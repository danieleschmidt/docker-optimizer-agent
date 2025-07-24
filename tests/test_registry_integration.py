"""Tests for Docker registry integration functionality."""

from unittest.mock import Mock, patch

import pytest

from docker_optimizer.registry_integration import (
    RegistryComparison,
    RegistryIntegrator,
    RegistryRecommendation,
    RegistryVulnerabilityData,
)


class TestRegistryIntegrator:
    """Test Docker registry integration functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.integrator = RegistryIntegrator()

    def test_ecr_vulnerability_scan(self):
        """Test pulling vulnerability data from AWS ECR."""
        image_name = "my-app:latest"

        vulnerabilities = self.integrator.scan_ecr_vulnerabilities(image_name)

        assert isinstance(vulnerabilities, RegistryVulnerabilityData)
        assert vulnerabilities.registry_type == "ECR"
        assert vulnerabilities.image_name == image_name
        assert hasattr(vulnerabilities, 'critical_count')
        assert hasattr(vulnerabilities, 'high_count')
        assert hasattr(vulnerabilities, 'medium_count')
        assert hasattr(vulnerabilities, 'low_count')

    def test_acr_vulnerability_scan(self):
        """Test pulling vulnerability data from Azure ACR."""
        image_name = "my-app:latest"

        vulnerabilities = self.integrator.scan_acr_vulnerabilities(image_name)

        assert isinstance(vulnerabilities, RegistryVulnerabilityData)
        assert vulnerabilities.registry_type == "ACR"
        assert vulnerabilities.image_name == image_name

    def test_gcr_vulnerability_scan(self):
        """Test pulling vulnerability data from Google GCR."""
        image_name = "my-app:latest"

        vulnerabilities = self.integrator.scan_gcr_vulnerabilities(image_name)

        assert isinstance(vulnerabilities, RegistryVulnerabilityData)
        assert vulnerabilities.registry_type == "GCR"
        assert vulnerabilities.image_name == image_name

    def test_compare_across_registries(self):
        """Test comparing the same image across multiple registries."""
        image_name = "my-app:latest"
        registries = ["ECR", "ACR", "GCR"]

        comparison = self.integrator.compare_across_registries(image_name, registries)

        assert isinstance(comparison, RegistryComparison)
        assert comparison.image_name == image_name
        assert len(comparison.registry_data) == len(registries)
        assert hasattr(comparison, 'best_registry')
        assert hasattr(comparison, 'vulnerability_score')

    def test_get_registry_recommendations(self):
        """Test getting registry-specific optimization recommendations."""
        dockerfile_content = """
FROM node:16-alpine
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
"""

        recommendations = self.integrator.get_registry_recommendations(dockerfile_content)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, RegistryRecommendation) for rec in recommendations)

    def test_registry_specific_base_images(self):
        """Test recommendations for registry-specific base images."""
        language = "python"

        recommendations = self.integrator.get_registry_base_image_recommendations(language)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should recommend images from different registries
        registry_types = {rec.registry_type for rec in recommendations}
        assert len(registry_types) > 1

    @patch('docker_optimizer.registry_integration.requests.get')
    def test_ecr_api_integration(self, mock_get):
        """Test ECR API integration with mocked responses."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'imageScanFindingsSummary': {
                'findingCounts': {
                    'CRITICAL': 2,
                    'HIGH': 5,
                    'MEDIUM': 10,
                    'LOW': 15
                }
            }
        }
        mock_get.return_value = mock_response

        result = self.integrator.scan_ecr_vulnerabilities("test-image:latest")

        assert result.critical_count == 2
        assert result.high_count == 5
        assert result.medium_count == 10
        assert result.low_count == 15

    def test_error_handling_invalid_registry(self):
        """Test error handling for invalid registry types."""
        with pytest.raises(ValueError, match="Unsupported registry type"):
            self.integrator.scan_registry_vulnerabilities("invalid-registry", "image:latest")

    def test_registry_availability_check(self):
        """Test checking if registries are available and accessible."""
        availability = self.integrator.check_registry_availability(["ECR", "ACR", "GCR"])

        assert isinstance(availability, dict)
        assert all(registry in availability for registry in ["ECR", "ACR", "GCR"])
        assert all(isinstance(status, bool) for status in availability.values())
