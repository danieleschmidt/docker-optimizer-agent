"""Test data validation and fixture integrity tests.

This module ensures that test fixtures and data are valid and consistent.
"""

import json
import yaml
from pathlib import Path
import pytest


class TestFixtureIntegrity:
    """Test that all test fixtures are valid and usable."""
    
    @pytest.fixture
    def fixtures_dir(self):
        """Get fixtures directory path."""
        return Path(__file__).parent / "fixtures"
    
    def test_dockerfile_fixtures_valid(self, fixtures_dir):
        """Test that all Dockerfile fixtures are valid."""
        dockerfile_dir = fixtures_dir / "dockerfiles"
        
        if not dockerfile_dir.exists():
            pytest.skip("No dockerfile fixtures directory")
        
        dockerfile_files = list(dockerfile_dir.glob("*.dockerfile"))
        assert len(dockerfile_files) > 0, "No dockerfile fixtures found"
        
        for dockerfile_path in dockerfile_files:
            content = dockerfile_path.read_text()
            
            # Basic validation
            assert content.strip(), f"Dockerfile {dockerfile_path.name} is empty"
            assert "FROM" in content.upper(), f"Dockerfile {dockerfile_path.name} has no FROM instruction"
            
            # Check for common Dockerfile instructions
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            from_lines = [line for line in lines if line.upper().startswith('FROM')]
            assert len(from_lines) >= 1, f"Dockerfile {dockerfile_path.name} has no FROM instruction"
    
    def test_json_fixtures_valid(self, fixtures_dir):
        """Test that all JSON fixtures are valid."""
        json_files = list(fixtures_dir.rglob("*.json"))
        
        for json_path in json_files:
            try:
                content = json_path.read_text()
                json.loads(content)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in {json_path}: {e}")
    
    def test_yaml_fixtures_valid(self, fixtures_dir):
        """Test that all YAML fixtures are valid."""
        yaml_files = list(fixtures_dir.rglob("*.yml")) + list(fixtures_dir.rglob("*.yaml"))
        
        for yaml_path in yaml_files:
            try:
                content = yaml_path.read_text()
                yaml.safe_load(content)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in {yaml_path}: {e}")
    
    def test_security_fixtures_structure(self, fixtures_dir):
        """Test that security fixtures have expected structure."""
        security_dir = fixtures_dir / "security"
        
        if not security_dir.exists():
            pytest.skip("No security fixtures directory")
        
        # Test Trivy report structure
        trivy_report = security_dir / "trivy-report.json"
        if trivy_report.exists():
            content = json.loads(trivy_report.read_text())
            
            # Validate expected Trivy report structure
            assert "SchemaVersion" in content, "Trivy report missing SchemaVersion"
            assert "Results" in content, "Trivy report missing Results"
            
            if content["Results"]:
                result = content["Results"][0]
                assert "Target" in result, "Trivy result missing Target"
                
                if "Vulnerabilities" in result:
                    vuln = result["Vulnerabilities"][0]
                    required_fields = ["VulnerabilityID", "Severity", "PkgName"]
                    for field in required_fields:
                        assert field in vuln, f"Vulnerability missing {field}"
    
    def test_performance_fixtures_valid(self, fixtures_dir):
        """Test that performance fixtures are valid."""
        performance_dir = fixtures_dir / "performance"
        
        if not performance_dir.exists():
            pytest.skip("No performance fixtures directory")
        
        # Add validation for performance fixture files
        # This can be expanded as performance fixtures are added
    
    def test_config_fixtures_loadable(self, fixtures_dir):
        """Test that configuration fixtures can be loaded."""
        configs_dir = fixtures_dir / "configs"
        
        if not configs_dir.exists():
            pytest.skip("No config fixtures directory")
        
        config_files = list(configs_dir.glob("*.yml")) + list(configs_dir.glob("*.yaml"))
        
        for config_path in config_files:
            config_data = yaml.safe_load(config_path.read_text())
            
            # Basic configuration validation
            assert isinstance(config_data, dict), f"Config {config_path.name} is not a dictionary"
            
            # Check for common configuration sections
            if "environment" in config_data:
                assert isinstance(config_data["environment"], str)
            
            if "logging" in config_data:
                assert isinstance(config_data["logging"], dict)


class TestTestDataConsistency:
    """Test that test data is consistent across different test files."""
    
    def test_sample_dockerfiles_consistent(self, sample_dockerfile, complex_dockerfile):
        """Test that sample dockerfiles from fixtures are consistent."""
        # Both should be valid dockerfile content
        assert "FROM" in sample_dockerfile.upper()
        assert "FROM" in complex_dockerfile.upper()
        
        # Complex dockerfile should actually be more complex
        simple_lines = len([l for l in sample_dockerfile.split('\n') if l.strip()])
        complex_lines = len([l for l in complex_dockerfile.split('\n') if l.strip()])
        assert complex_lines > simple_lines, "Complex dockerfile should have more instructions"
    
    def test_security_test_data_realistic(self, dockerfile_with_security_issues):
        """Test that security test data contains realistic issues."""
        content = dockerfile_with_security_issues.upper()
        
        # Should contain common security issues
        security_issues = [
            "USER ROOT",  # Running as root
            "FROM UBUNTU:LATEST",  # Using latest tag
            "EXPOSE 22",  # Exposing SSH port
        ]
        
        found_issues = sum(1 for issue in security_issues if issue in content)
        assert found_issues >= 2, "Security test dockerfile should contain multiple realistic issues"


class TestMockDataValidity:
    """Test that mock data structures match expected formats."""
    
    def test_mock_trivy_output_structure(self, mock_trivy_output):
        """Test that mock Trivy output has correct structure."""
        assert "Results" in mock_trivy_output
        assert isinstance(mock_trivy_output["Results"], list)
        
        if mock_trivy_output["Results"]:
            result = mock_trivy_output["Results"][0]
            assert "Target" in result
            assert "Vulnerabilities" in result
            
            if result["Vulnerabilities"]:
                vuln = result["Vulnerabilities"][0]
                required_fields = ["VulnerabilityID", "Severity", "PkgName"]
                for field in required_fields:
                    assert field in vuln
    
    def test_mock_optimization_result_structure(self, mock_optimization_result):
        """Test that mock optimization result has correct structure."""
        required_fields = [
            "original_dockerfile",
            "optimized_dockerfile", 
            "size_reduction",
            "security_score",
            "explanation",
            "recommendations"
        ]
        
        for field in required_fields:
            assert field in mock_optimization_result
        
        # Validate data types
        assert isinstance(mock_optimization_result["size_reduction"], (int, float))
        assert isinstance(mock_optimization_result["security_score"], dict)
        assert isinstance(mock_optimization_result["recommendations"], list)
        assert "grade" in mock_optimization_result["security_score"]