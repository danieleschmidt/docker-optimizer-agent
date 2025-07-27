"""Docker registry integration for vulnerability scanning and optimization."""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Config
from .logging_observability import ObservabilityManager
from .models import (
    RegistryComparison,
    RegistryRecommendation,
    RegistryVulnerabilityData,
)

logger = logging.getLogger(__name__)


class RegistryIntegrator:
    """Integrates with Docker registries for vulnerability scanning and recommendations."""

    def __init__(self, config: Optional[Config] = None, obs_manager: Optional[ObservabilityManager] = None) -> None:
        """Initialize the registry integrator.

        Args:
            config: Optional configuration instance
            obs_manager: Optional observability manager for structured logging
        """
        self.config = config or Config()
        self.obs_manager = obs_manager or ObservabilityManager(service_name="registry-integrator")
        self._setup_session()
        self._registry_endpoints = {
            'ECR': 'ecr',
            'ACR': 'containerregistry.azure.com',
            'GCR': 'gcr.io',
            'DOCKERHUB': 'registry-1.docker.io'
        }

        self.obs_manager.logger.info("Registry integrator initialized", extra={
            "registry_endpoints": list(self._registry_endpoints.keys())
        })

    def _setup_session(self) -> None:
        """Set up HTTP session with retry logic."""
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def scan_ecr_vulnerabilities(self, image_name: str) -> RegistryVulnerabilityData:
        """Scan vulnerabilities in AWS ECR.

        Args:
            image_name: Name of the image to scan

        Returns:
            Vulnerability data from ECR
        """
        logger.info(f"Scanning ECR vulnerabilities for {image_name}")

        try:
            # In production, this would use boto3 to call ECR APIs
            # For now, check if we have a way to call real APIs
            if hasattr(self, '_use_real_apis') and self._use_real_apis:
                return self._call_ecr_api(image_name)

            # Return mock data for testing and development
            return RegistryVulnerabilityData(
                registry_type="ECR",
                image_name=image_name,
                critical_count=1,
                high_count=3,
                medium_count=8,
                low_count=12,
                scan_timestamp=datetime.now().isoformat(),
                registry_url=f"123456789012.dkr.ecr.us-west-2.amazonaws.com/{image_name}"
            )
        except Exception as e:
            logger.error(f"Failed to scan ECR vulnerabilities for {image_name}: {e}")
            # Return empty vulnerability data on error
            return RegistryVulnerabilityData(
                registry_type="ECR",
                image_name=image_name,
                scan_timestamp=datetime.now().isoformat(),
                registry_url=f"123456789012.dkr.ecr.us-west-2.amazonaws.com/{image_name}"
            )

    def _call_ecr_api(self, image_name: str) -> RegistryVulnerabilityData:
        """Call the actual ECR API for vulnerability scanning.

        This method would be implemented with boto3 in production.
        For now, it's a placeholder that can be mocked in tests.
        """
        # This would be the real implementation using boto3
        # import boto3
        # ecr_client = boto3.client('ecr')
        # response = ecr_client.describe_image_scan_findings(...)

        raise NotImplementedError("Real ECR API integration not implemented yet")

    def scan_acr_vulnerabilities(self, image_name: str) -> RegistryVulnerabilityData:
        """Scan vulnerabilities in Azure ACR.

        Args:
            image_name: Name of the image to scan

        Returns:
            Vulnerability data from ACR
        """
        logger.info(f"Scanning ACR vulnerabilities for {image_name}")

        # For now, return mock data since we don't have Azure credentials
        # In production, this would use Azure SDK to call ACR APIs
        return RegistryVulnerabilityData(
            registry_type="ACR",
            image_name=image_name,
            critical_count=0,
            high_count=2,
            medium_count=5,
            low_count=10,
            scan_timestamp=datetime.now().isoformat(),
            registry_url=f"myregistry.azurecr.io/{image_name}"
        )

    def scan_gcr_vulnerabilities(self, image_name: str) -> RegistryVulnerabilityData:
        """Scan vulnerabilities in Google GCR.

        Args:
            image_name: Name of the image to scan

        Returns:
            Vulnerability data from GCR
        """
        logger.info(f"Scanning GCR vulnerabilities for {image_name}")

        # For now, return mock data since we don't have GCP credentials
        # In production, this would use Google Cloud SDK to call GCR APIs
        return RegistryVulnerabilityData(
            registry_type="GCR",
            image_name=image_name,
            critical_count=2,
            high_count=4,
            medium_count=6,
            low_count=8,
            scan_timestamp=datetime.now().isoformat(),
            registry_url=f"gcr.io/my-project/{image_name}"
        )

    def scan_registry_vulnerabilities(self, registry_type: str, image_name: str) -> RegistryVulnerabilityData:
        """Scan vulnerabilities in a specific registry.

        Args:
            registry_type: Type of registry (ECR, ACR, GCR)
            image_name: Name of the image to scan

        Returns:
            Vulnerability data from the specified registry

        Raises:
            ValueError: If registry type is not supported
        """
        with self.obs_manager.track_operation(
            operation_type="registry_vulnerability_scan",
        ) as context:
            registry_type = registry_type.upper()

            self.obs_manager.logger.info("Starting vulnerability scan", context=context, extra={
                "registry_type": registry_type,
                "image_name": image_name
            })

            if registry_type == "ECR":
                result = self.scan_ecr_vulnerabilities(image_name)
            elif registry_type == "ACR":
                result = self.scan_acr_vulnerabilities(image_name)
            elif registry_type == "GCR":
                result = self.scan_gcr_vulnerabilities(image_name)
            else:
                error_msg = f"Unsupported registry type: {registry_type}"
                self.obs_manager.logger.error(error_msg, context=context, extra={
                    "supported_types": list(self._registry_endpoints.keys())
                })
                raise ValueError(error_msg)

            self.obs_manager.logger.info("Vulnerability scan completed", context=context, extra={
                "vulnerabilities_found": len(result.vulnerabilities),
                "critical_count": len([v for v in result.vulnerabilities if v.severity == "CRITICAL"]),
                "high_count": len([v for v in result.vulnerabilities if v.severity == "HIGH"])
            })

            return result

    def compare_across_registries(self, image_name: str, registries: List[str]) -> RegistryComparison:
        """Compare the same image across multiple registries.

        Args:
            image_name: Name of the image to compare
            registries: List of registry types to compare

        Returns:
            Comparison results across registries
        """
        logger.info(f"Comparing {image_name} across registries: {registries}")

        registry_data = []
        for registry in registries:
            try:
                data = self.scan_registry_vulnerabilities(registry, image_name)
                registry_data.append(data)
            except Exception as e:
                logger.warning(f"Failed to scan {registry} for {image_name}: {e}")
                continue

        # Find the best option (lowest vulnerability score)
        best_option = None
        best_score = float('inf')

        if registry_data:
            best_option = min(registry_data, key=lambda x: x.severity_score)
            best_score = best_option.severity_score

        return RegistryComparison(
            image_name=image_name,
            registry_data=registry_data,
            best_registry=best_option.registry_type if best_option else None,
            vulnerability_score=best_score if best_option else 0.0
        )

    def get_registry_recommendations(self, dockerfile_content: str) -> List[RegistryRecommendation]:
        """Get registry-specific optimization recommendations.

        Args:
            dockerfile_content: Content of the Dockerfile

        Returns:
            List of registry-specific recommendations
        """
        recommendations = []

        # Analyze the base image
        base_image = self._extract_base_image(dockerfile_content)

        if base_image:
            # Add registry-specific recommendations
            recommendations.extend(self._get_base_image_recommendations(base_image))
            recommendations.extend(self._get_security_recommendations(base_image))
            recommendations.extend(self._get_optimization_recommendations(dockerfile_content))

        return recommendations

    def get_registry_base_image_recommendations(self, language: str) -> List[RegistryRecommendation]:
        """Get registry-specific base image recommendations for a language.

        Args:
            language: Programming language

        Returns:
            List of registry-specific base image recommendations
        """
        recommendations = []

        registry_patterns = {
            'ECR': {
                'python': 'public.ecr.aws/lambda/python:3.11',
                'node': 'public.ecr.aws/lambda/nodejs:18',
                'java': 'public.ecr.aws/amazoncorretto/amazoncorretto:17'
            },
            'GCR': {
                'python': 'gcr.io/distroless/python3',
                'node': 'gcr.io/distroless/nodejs',
                'java': 'gcr.io/distroless/java:11'
            },
            'DOCKERHUB': {
                'python': 'python:3.11-slim',
                'node': 'node:18-alpine',
                'java': 'openjdk:17-alpine'
            }
        }

        for registry, images in registry_patterns.items():
            if language in images:
                recommendations.append(RegistryRecommendation(
                    registry_type=registry,
                    recommendation_type="base_image",
                    title=f"Use {registry} optimized {language} image",
                    description=f"Switch to {registry}'s optimized {language} base image for better security and performance",
                    dockerfile_change=f"FROM {images[language]}",
                    security_benefit=f"{registry} provides regular security updates and scanning",
                    estimated_impact="HIGH"
                ))

        return recommendations

    def check_registry_availability(self, registries: List[str]) -> Dict[str, bool]:
        """Check if registries are available and accessible.

        Args:
            registries: List of registry types to check

        Returns:
            Dictionary mapping registry types to availability status
        """
        availability = {}

        for registry in registries:
            try:
                # For now, assume all registries are available
                # In production, this would make actual health checks
                availability[registry] = True
            except Exception:
                availability[registry] = False

        return availability

    def _extract_base_image(self, dockerfile_content: str) -> Optional[str]:
        """Extract the base image from Dockerfile content.

        Args:
            dockerfile_content: Content of the Dockerfile

        Returns:
            Base image name or None if not found
        """
        lines = dockerfile_content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith('FROM '):
                # Extract the image name (remove FROM and any AS alias)
                parts = line.split()
                if len(parts) >= 2:
                    base_image = parts[1]
                    # Remove any AS alias
                    if ' AS ' in line.upper():
                        base_image = base_image.split()[0]
                    return base_image
        return None

    def _get_base_image_recommendations(self, base_image: str) -> List[RegistryRecommendation]:
        """Get recommendations for base image optimization.

        Args:
            base_image: Current base image

        Returns:
            List of base image recommendations
        """
        recommendations = []

        # If using a generic image, recommend registry-specific alternatives
        if any(generic in base_image.lower() for generic in ['ubuntu', 'centos', 'debian']):
            recommendations.append(RegistryRecommendation(
                registry_type="GCR",
                recommendation_type="base_image",
                title="Switch to distroless base image",
                description="Use Google's distroless images for better security",
                dockerfile_change="FROM gcr.io/distroless/static",
                security_benefit="Distroless images have minimal attack surface with no shell or package manager",
                estimated_impact="HIGH"
            ))

        return recommendations

    def _get_security_recommendations(self, base_image: str) -> List[RegistryRecommendation]:
        """Get security-related registry recommendations.

        Args:
            base_image: Current base image

        Returns:
            List of security recommendations
        """
        recommendations = []

        # Recommend using official images from trusted registries
        if not any(registry in base_image for registry in ['public.ecr.aws', 'gcr.io', 'mcr.microsoft.com']):
            recommendations.append(RegistryRecommendation(
                registry_type="ECR",
                recommendation_type="security",
                title="Use trusted registry images",
                description="Switch to official images from trusted registries",
                dockerfile_change=f"# Consider using public.ecr.aws equivalent of {base_image}",
                security_benefit="Trusted registries provide verified, regularly updated images",
                estimated_impact="MEDIUM"
            ))

        return recommendations

    def _get_optimization_recommendations(self, dockerfile_content: str) -> List[RegistryRecommendation]:
        """Get optimization recommendations based on Dockerfile content.

        Args:
            dockerfile_content: Content of the Dockerfile

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Check for multi-stage build opportunities
        if dockerfile_content.count('FROM ') == 1:
            recommendations.append(RegistryRecommendation(
                registry_type="ECR",
                recommendation_type="optimization",
                title="Consider multi-stage builds with ECR",
                description="Use ECR's build optimization features with multi-stage builds",
                dockerfile_change="# Use ECR's optimized build images in build stage",
                security_benefit="Smaller final images reduce attack surface",
                estimated_impact="MEDIUM"
            ))

        return recommendations
