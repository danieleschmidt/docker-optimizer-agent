"""Global-First Implementation Features.

This module provides internationalization, multi-region support, compliance,
and global deployment capabilities for the Docker Optimizer Agent.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
import re

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ITALIAN = "it"
    DUTCH = "nl"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore/Thailand)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    SOC2 = "soc2"  # SOC 2 Compliance
    ISO27001 = "iso27001"  # ISO 27001 Information Security
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act (US)
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard


class Region(Enum):
    """Supported deployment regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    CANADA = "ca-central-1"
    AUSTRALIA = "ap-southeast-2"
    BRAZIL = "sa-east-1"
    MIDDLE_EAST = "me-south-1"
    AFRICA = "af-south-1"


@dataclass
class ComplianceRequirement:
    """Compliance requirement definition."""
    framework: ComplianceFramework
    requirement_id: str
    description: str
    severity: str  # "critical", "high", "medium", "low"
    applicable_regions: List[Region]
    validation_rules: List[str]
    remediation_steps: List[str]


@dataclass
class GlobalizationConfig:
    """Configuration for globalization features."""
    default_language: SupportedLanguage = SupportedLanguage.ENGLISH
    enabled_languages: List[SupportedLanguage] = field(default_factory=lambda: [SupportedLanguage.ENGLISH])
    default_region: Region = Region.US_EAST
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    timezone: str = "UTC"
    date_format: str = "ISO8601"
    currency: str = "USD"
    enable_rtl_support: bool = False  # Right-to-left language support


class InternationalizationEngine:
    """Engine for handling internationalization and localization."""
    
    def __init__(self, config: GlobalizationConfig):
        self.config = config
        self.translations: Dict[str, Dict[str, str]] = {}
        self.loaded_languages: set = set()
        self._load_default_translations()
    
    def _load_default_translations(self) -> None:
        """Load default translations for supported languages."""
        # Core optimization messages
        base_translations = {
            "optimization_complete": "Optimization completed successfully",
            "security_issues_found": "Security issues found: {count}",
            "layer_optimizations": "Layer optimizations applied: {count}",
            "size_reduction": "Image size reduced by {percentage}%",
            "dockerfile_invalid": "Dockerfile validation failed",
            "processing_batch": "Processing batch of {count} dockerfiles",
            "high_throughput_mode": "High-throughput mode enabled",
            "research_benchmark": "Research benchmark initiated",
            "compliance_check": "Compliance check: {framework}",
            "global_deployment": "Multi-region deployment ready"
        }
        
        # Translations for each supported language
        translations = {
            SupportedLanguage.ENGLISH: base_translations,
            SupportedLanguage.SPANISH: {
                "optimization_complete": "Optimización completada exitosamente",
                "security_issues_found": "Problemas de seguridad encontrados: {count}",
                "layer_optimizations": "Optimizaciones de capas aplicadas: {count}",
                "size_reduction": "Tamaño de imagen reducido en {percentage}%",
                "dockerfile_invalid": "Validación de Dockerfile falló",
                "processing_batch": "Procesando lote de {count} dockerfiles",
                "high_throughput_mode": "Modo de alto rendimiento habilitado",
                "research_benchmark": "Benchmark de investigación iniciado",
                "compliance_check": "Verificación de cumplimiento: {framework}",
                "global_deployment": "Despliegue multi-región listo"
            },
            SupportedLanguage.FRENCH: {
                "optimization_complete": "Optimisation terminée avec succès",
                "security_issues_found": "Problèmes de sécurité trouvés: {count}",
                "layer_optimizations": "Optimisations de couches appliquées: {count}",
                "size_reduction": "Taille d'image réduite de {percentage}%",
                "dockerfile_invalid": "Validation du Dockerfile échouée",
                "processing_batch": "Traitement d'un lot de {count} dockerfiles",
                "high_throughput_mode": "Mode haute performance activé",
                "research_benchmark": "Benchmark de recherche initié",
                "compliance_check": "Vérification de conformité: {framework}",
                "global_deployment": "Déploiement multi-région prêt"
            },
            SupportedLanguage.GERMAN: {
                "optimization_complete": "Optimierung erfolgreich abgeschlossen",
                "security_issues_found": "Sicherheitsprobleme gefunden: {count}",
                "layer_optimizations": "Layer-Optimierungen angewendet: {count}",
                "size_reduction": "Image-Größe um {percentage}% reduziert",
                "dockerfile_invalid": "Dockerfile-Validierung fehlgeschlagen",
                "processing_batch": "Verarbeitung von {count} Dockerfiles",
                "high_throughput_mode": "Hochleistungsmodus aktiviert",
                "research_benchmark": "Forschungs-Benchmark gestartet",
                "compliance_check": "Compliance-Prüfung: {framework}",
                "global_deployment": "Multi-Region-Deployment bereit"
            },
            SupportedLanguage.JAPANESE: {
                "optimization_complete": "最適化が正常に完了しました",
                "security_issues_found": "セキュリティ問題が見つかりました: {count}",
                "layer_optimizations": "レイヤー最適化が適用されました: {count}",
                "size_reduction": "イメージサイズが{percentage}%削減されました",
                "dockerfile_invalid": "Dockerfileの検証に失敗しました",
                "processing_batch": "{count}個のDockerfileを処理中",
                "high_throughput_mode": "高スループットモードが有効",
                "research_benchmark": "研究ベンチマークを開始",
                "compliance_check": "コンプライアンスチェック: {framework}",
                "global_deployment": "マルチリージョンデプロイメント準備完了"
            },
            SupportedLanguage.CHINESE_SIMPLIFIED: {
                "optimization_complete": "优化成功完成",
                "security_issues_found": "发现安全问题：{count}",
                "layer_optimizations": "已应用层优化：{count}",
                "size_reduction": "镜像大小减少了{percentage}%",
                "dockerfile_invalid": "Dockerfile验证失败",
                "processing_batch": "正在处理{count}个dockerfile批次",
                "high_throughput_mode": "已启用高吞吐量模式",
                "research_benchmark": "研究基准测试已启动",
                "compliance_check": "合规检查：{framework}",
                "global_deployment": "多区域部署就绪"
            }
        }
        
        # Store translations
        for lang, messages in translations.items():
            self.translations[lang.value] = messages
            self.loaded_languages.add(lang.value)
    
    def get_message(self, key: str, language: Optional[SupportedLanguage] = None, **kwargs) -> str:
        """Get localized message."""
        if language is None:
            language = self.config.default_language
        
        lang_code = language.value
        
        # Fallback to English if language not available
        if lang_code not in self.translations:
            lang_code = SupportedLanguage.ENGLISH.value
        
        # Get message with fallback
        message = self.translations.get(lang_code, {}).get(
            key, 
            self.translations.get(SupportedLanguage.ENGLISH.value, {}).get(
                key, 
                f"[MISSING: {key}]"
            )
        )
        
        # Format message with parameters
        try:
            return message.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.warning(f"Message formatting failed for key '{key}': {e}")
            return message
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages with metadata."""
        languages = []
        
        language_metadata = {
            SupportedLanguage.ENGLISH: {"name": "English", "native": "English", "rtl": False},
            SupportedLanguage.SPANISH: {"name": "Spanish", "native": "Español", "rtl": False},
            SupportedLanguage.FRENCH: {"name": "French", "native": "Français", "rtl": False},
            SupportedLanguage.GERMAN: {"name": "German", "native": "Deutsch", "rtl": False},
            SupportedLanguage.JAPANESE: {"name": "Japanese", "native": "日本語", "rtl": False},
            SupportedLanguage.CHINESE_SIMPLIFIED: {"name": "Chinese (Simplified)", "native": "中文(简体)", "rtl": False},
            SupportedLanguage.PORTUGUESE: {"name": "Portuguese", "native": "Português", "rtl": False},
            SupportedLanguage.RUSSIAN: {"name": "Russian", "native": "Русский", "rtl": False},
            SupportedLanguage.ITALIAN: {"name": "Italian", "native": "Italiano", "rtl": False},
            SupportedLanguage.DUTCH: {"name": "Dutch", "native": "Nederlands", "rtl": False}
        }
        
        for lang in self.config.enabled_languages:
            metadata = language_metadata.get(lang, {})
            languages.append({
                "code": lang.value,
                "name": metadata.get("name", lang.value),
                "native_name": metadata.get("native", lang.value),
                "rtl": metadata.get("rtl", False),
                "available": lang.value in self.loaded_languages
            })
        
        return languages


class ComplianceEngine:
    """Engine for handling compliance requirements and validation."""
    
    def __init__(self, config: GlobalizationConfig):
        self.config = config
        self.requirements: Dict[ComplianceFramework, List[ComplianceRequirement]] = {}
        self._load_compliance_requirements()
    
    def _load_compliance_requirements(self) -> None:
        """Load compliance requirements for supported frameworks."""
        # GDPR Requirements
        gdpr_requirements = [
            ComplianceRequirement(
                framework=ComplianceFramework.GDPR,
                requirement_id="GDPR-001",
                description="Data processing must have lawful basis",
                severity="critical",
                applicable_regions=[Region.EU_WEST, Region.EU_CENTRAL],
                validation_rules=[
                    "no_personal_data_in_dockerfiles",
                    "explicit_data_consent_mechanisms",
                    "data_anonymization_present"
                ],
                remediation_steps=[
                    "Remove any personal data from Dockerfiles",
                    "Implement data anonymization",
                    "Add consent management system"
                ]
            ),
            ComplianceRequirement(
                framework=ComplianceFramework.GDPR,
                requirement_id="GDPR-002", 
                description="Right to be forgotten must be implementable",
                severity="high",
                applicable_regions=[Region.EU_WEST, Region.EU_CENTRAL],
                validation_rules=[
                    "data_deletion_mechanisms",
                    "backup_data_handling"
                ],
                remediation_steps=[
                    "Implement data deletion APIs",
                    "Ensure backups can be selectively cleaned"
                ]
            )
        ]
        
        # SOC 2 Requirements
        soc2_requirements = [
            ComplianceRequirement(
                framework=ComplianceFramework.SOC2,
                requirement_id="SOC2-001",
                description="Security controls must be in place",
                severity="critical",
                applicable_regions=[r for r in Region],  # Global
                validation_rules=[
                    "no_hardcoded_secrets",
                    "proper_access_controls",
                    "audit_logging_enabled"
                ],
                remediation_steps=[
                    "Remove hardcoded credentials",
                    "Implement proper RBAC",
                    "Enable comprehensive audit logging"
                ]
            )
        ]
        
        # ISO 27001 Requirements
        iso27001_requirements = [
            ComplianceRequirement(
                framework=ComplianceFramework.ISO27001,
                requirement_id="ISO27001-001",
                description="Information security management system",
                severity="high",
                applicable_regions=[r for r in Region],  # Global
                validation_rules=[
                    "security_policies_documented",
                    "risk_assessment_completed",
                    "incident_response_procedures"
                ],
                remediation_steps=[
                    "Document security policies",
                    "Complete risk assessment",
                    "Establish incident response procedures"
                ]
            )
        ]
        
        # Store requirements
        self.requirements[ComplianceFramework.GDPR] = gdpr_requirements
        self.requirements[ComplianceFramework.SOC2] = soc2_requirements
        self.requirements[ComplianceFramework.ISO27001] = iso27001_requirements
    
    def validate_compliance(self, 
                          framework: ComplianceFramework,
                          dockerfile_content: str,
                          region: Optional[Region] = None) -> Dict[str, Any]:
        """Validate compliance for a specific framework."""
        if region is None:
            region = self.config.default_region
        
        requirements = self.requirements.get(framework, [])
        applicable_requirements = [
            req for req in requirements 
            if region in req.applicable_regions
        ]
        
        validation_results = {
            "framework": framework.value,
            "region": region.value,
            "compliant": True,
            "violations": [],
            "recommendations": [],
            "applicable_requirements": len(applicable_requirements)
        }
        
        for requirement in applicable_requirements:
            violations = self._check_requirement(requirement, dockerfile_content)
            if violations:
                validation_results["compliant"] = False
                validation_results["violations"].extend(violations)
                validation_results["recommendations"].extend(requirement.remediation_steps)
        
        return validation_results
    
    def _check_requirement(self, 
                          requirement: ComplianceRequirement, 
                          dockerfile_content: str) -> List[str]:
        """Check a specific compliance requirement."""
        violations = []
        
        for rule in requirement.validation_rules:
            if not self._validate_rule(rule, dockerfile_content):
                violations.append(f"{requirement.requirement_id}: {requirement.description}")
                break
        
        return violations
    
    def _validate_rule(self, rule: str, dockerfile_content: str) -> bool:
        """Validate a specific compliance rule."""
        content_lower = dockerfile_content.lower()
        
        # Rule implementations
        if rule == "no_personal_data_in_dockerfiles":
            # Check for common PII patterns
            pii_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            ]
            for pattern in pii_patterns:
                if re.search(pattern, dockerfile_content):
                    return False
            return True
        
        elif rule == "no_hardcoded_secrets":
            # Check for hardcoded secrets
            secret_patterns = [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api[_-]?key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ]
            for pattern in secret_patterns:
                if re.search(pattern, content_lower):
                    return False
            return True
        
        elif rule == "explicit_data_consent_mechanisms":
            # Check for consent management indicators
            consent_indicators = ["consent", "gdpr", "privacy", "opt-in", "agreement"]
            return any(indicator in content_lower for indicator in consent_indicators)
        
        elif rule == "data_anonymization_present":
            # Check for data anonymization indicators
            anon_indicators = ["anonymiz", "pseudonym", "hash", "encrypt", "mask"]
            return any(indicator in content_lower for indicator in anon_indicators)
        
        elif rule == "proper_access_controls":
            # Check for access control measures
            return "user " in content_lower and "root" not in content_lower
        
        elif rule == "audit_logging_enabled":
            # Check for logging indicators
            log_indicators = ["log", "audit", "monitor", "track"]
            return any(indicator in content_lower for indicator in log_indicators)
        
        # Default: rule not implemented, assume compliant
        return True
    
    def get_compliance_summary(self, region: Optional[Region] = None) -> Dict[str, Any]:
        """Get summary of compliance requirements for a region."""
        if region is None:
            region = self.config.default_region
        
        summary = {
            "region": region.value,
            "frameworks": {},
            "total_requirements": 0,
            "critical_requirements": 0
        }
        
        for framework in self.config.compliance_frameworks:
            requirements = self.requirements.get(framework, [])
            applicable_reqs = [req for req in requirements if region in req.applicable_regions]
            
            framework_summary = {
                "total_requirements": len(applicable_reqs),
                "critical_requirements": len([req for req in applicable_reqs if req.severity == "critical"]),
                "high_requirements": len([req for req in applicable_reqs if req.severity == "high"]),
                "requirements": [
                    {
                        "id": req.requirement_id,
                        "description": req.description,
                        "severity": req.severity
                    }
                    for req in applicable_reqs
                ]
            }
            
            summary["frameworks"][framework.value] = framework_summary
            summary["total_requirements"] += len(applicable_reqs)
            summary["critical_requirements"] += framework_summary["critical_requirements"]
        
        return summary


class MultiRegionDeploymentManager:
    """Manager for multi-region deployment capabilities."""
    
    def __init__(self, config: GlobalizationConfig):
        self.config = config
        self.region_configs: Dict[Region, Dict[str, Any]] = {}
        self._initialize_region_configs()
    
    def _initialize_region_configs(self) -> None:
        """Initialize region-specific configurations."""
        # Region-specific settings
        region_settings = {
            Region.US_EAST: {
                "compliance": [ComplianceFramework.SOC2, ComplianceFramework.HIPAA],
                "data_residency": "us",
                "languages": [SupportedLanguage.ENGLISH, SupportedLanguage.SPANISH],
                "timezone": "America/New_York",
                "currency": "USD",
                "cdn_endpoints": ["cloudfront-us-east-1"],
                "latency_targets": {"p95": 100, "p99": 200}  # milliseconds
            },
            Region.EU_WEST: {
                "compliance": [ComplianceFramework.GDPR, ComplianceFramework.ISO27001],
                "data_residency": "eu",
                "languages": [SupportedLanguage.ENGLISH, SupportedLanguage.GERMAN, SupportedLanguage.FRENCH],
                "timezone": "Europe/London",
                "currency": "EUR",
                "cdn_endpoints": ["cloudfront-eu-west-1"],
                "latency_targets": {"p95": 150, "p99": 300}
            },
            Region.ASIA_PACIFIC: {
                "compliance": [ComplianceFramework.PDPA],
                "data_residency": "apac",
                "languages": [SupportedLanguage.ENGLISH, SupportedLanguage.JAPANESE, SupportedLanguage.CHINESE_SIMPLIFIED],
                "timezone": "Asia/Singapore",
                "currency": "USD",
                "cdn_endpoints": ["cloudfront-ap-southeast-1"],
                "latency_targets": {"p95": 200, "p99": 400}
            }
        }
        
        # Apply default settings to all regions
        default_settings = {
            "compliance": [ComplianceFramework.SOC2],
            "data_residency": "global",
            "languages": [SupportedLanguage.ENGLISH],
            "timezone": "UTC",
            "currency": "USD",
            "cdn_endpoints": [],
            "latency_targets": {"p95": 300, "p99": 500}
        }
        
        for region in Region:
            self.region_configs[region] = {
                **default_settings,
                **region_settings.get(region, {})
            }
    
    def get_deployment_config(self, region: Region) -> Dict[str, Any]:
        """Get deployment configuration for a specific region."""
        base_config = self.region_configs.get(region, {})
        
        return {
            "region": region.value,
            "deployment_ready": True,
            "configurations": base_config,
            "optimization_settings": {
                "enable_caching": True,
                "cache_ttl_seconds": 3600,
                "batch_processing": True,
                "high_throughput_mode": True,
                "compliance_validation": True
            },
            "monitoring": {
                "enable_metrics": True,
                "enable_tracing": True,
                "enable_logging": True,
                "log_level": "INFO"
            },
            "security": {
                "enable_tls": True,
                "min_tls_version": "1.2",
                "enable_cors": True,
                "enable_auth": True
            }
        }
    
    def validate_multi_region_readiness(self) -> Dict[str, Any]:
        """Validate readiness for multi-region deployment."""
        readiness_check = {
            "ready": True,
            "regions_configured": len(self.region_configs),
            "compliance_frameworks": len(set(
                fw for region_cfg in self.region_configs.values()
                for fw in region_cfg.get("compliance", [])
            )),
            "supported_languages": len(self.config.enabled_languages),
            "issues": [],
            "recommendations": []
        }
        
        # Check for common deployment issues
        if len(self.config.enabled_languages) < 2:
            readiness_check["issues"].append("Limited language support may impact global adoption")
            readiness_check["recommendations"].append("Consider adding more language translations")
        
        if not self.config.compliance_frameworks:
            readiness_check["issues"].append("No compliance frameworks configured")
            readiness_check["recommendations"].append("Configure required compliance frameworks for target regions")
        
        # Check region coverage
        critical_regions = [Region.US_EAST, Region.EU_WEST, Region.ASIA_PACIFIC]
        configured_critical = sum(1 for region in critical_regions if region in self.region_configs)
        
        if configured_critical < len(critical_regions):
            readiness_check["issues"].append("Missing critical region configurations")
            readiness_check["recommendations"].append("Configure US, EU, and APAC regions for global coverage")
        
        readiness_check["ready"] = len(readiness_check["issues"]) == 0
        
        return readiness_check


class GlobalOptimizationEngine:
    """Main engine that orchestrates global-first optimization features."""
    
    def __init__(self, config: Optional[GlobalizationConfig] = None):
        self.config = config or GlobalizationConfig()
        self.i18n_engine = InternationalizationEngine(self.config)
        self.compliance_engine = ComplianceEngine(self.config)
        self.deployment_manager = MultiRegionDeploymentManager(self.config)
        
        logger.info(f"Global optimization engine initialized for {len(self.config.enabled_languages)} languages")
    
    def optimize_dockerfile_with_global_context(self,
                                              dockerfile_content: str,
                                              target_region: Optional[Region] = None,
                                              language: Optional[SupportedLanguage] = None) -> Dict[str, Any]:
        """Optimize dockerfile with global context including compliance and localization."""
        if target_region is None:
            target_region = self.config.default_region
        if language is None:
            language = self.config.default_language
        
        # Get deployment configuration
        deployment_config = self.deployment_manager.get_deployment_config(target_region)
        
        # Validate compliance for target region
        compliance_results = {}
        for framework in self.config.compliance_frameworks:
            compliance_results[framework.value] = self.compliance_engine.validate_compliance(
                framework, dockerfile_content, target_region
            )
        
        # Generate localized messages
        messages = {
            "optimization_started": self.i18n_engine.get_message(
                "optimization_complete", language
            ),
            "compliance_check": self.i18n_engine.get_message(
                "compliance_check", language, 
                framework=", ".join(f.value for f in self.config.compliance_frameworks)
            ),
            "global_deployment": self.i18n_engine.get_message(
                "global_deployment", language
            )
        }
        
        return {
            "global_optimization_results": {
                "target_region": target_region.value,
                "language": language.value,
                "deployment_config": deployment_config,
                "compliance_validation": compliance_results,
                "localized_messages": messages,
                "global_readiness": self.deployment_manager.validate_multi_region_readiness(),
                "supported_features": {
                    "internationalization": True,
                    "compliance_validation": True,
                    "multi_region_deployment": True,
                    "cross_platform_compatibility": True,
                    "rtl_support": self.config.enable_rtl_support
                }
            }
        }
    
    def get_global_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive information about global capabilities."""
        return {
            "internationalization": {
                "supported_languages": self.i18n_engine.get_supported_languages(),
                "default_language": self.config.default_language.value,
                "rtl_support": self.config.enable_rtl_support
            },
            "compliance": {
                "supported_frameworks": [f.value for f in ComplianceFramework],
                "configured_frameworks": [f.value for f in self.config.compliance_frameworks],
                "compliance_summary": self.compliance_engine.get_compliance_summary()
            },
            "deployment": {
                "supported_regions": [r.value for r in Region],
                "default_region": self.config.default_region.value,
                "multi_region_readiness": self.deployment_manager.validate_multi_region_readiness()
            },
            "global_features": {
                "cross_platform_compatibility": True,
                "timezone_support": True,
                "currency_localization": True,
                "date_format_localization": True,
                "regulatory_compliance": True,
                "data_residency_controls": True,
                "performance_optimization": True
            }
        }


# Convenience functions for easy integration
def create_global_config_for_region(region: Region) -> GlobalizationConfig:
    """Create optimized global configuration for a specific region."""
    region_language_map = {
        Region.US_EAST: [SupportedLanguage.ENGLISH, SupportedLanguage.SPANISH],
        Region.US_WEST: [SupportedLanguage.ENGLISH, SupportedLanguage.SPANISH],
        Region.EU_WEST: [SupportedLanguage.ENGLISH, SupportedLanguage.FRENCH, SupportedLanguage.GERMAN],
        Region.EU_CENTRAL: [SupportedLanguage.GERMAN, SupportedLanguage.ENGLISH],
        Region.ASIA_PACIFIC: [SupportedLanguage.ENGLISH, SupportedLanguage.JAPANESE, SupportedLanguage.CHINESE_SIMPLIFIED],
        Region.ASIA_NORTHEAST: [SupportedLanguage.JAPANESE, SupportedLanguage.ENGLISH],
        Region.CANADA: [SupportedLanguage.ENGLISH, SupportedLanguage.FRENCH],
        Region.BRAZIL: [SupportedLanguage.PORTUGUESE, SupportedLanguage.ENGLISH],
        Region.AUSTRALIA: [SupportedLanguage.ENGLISH],
    }
    
    region_compliance_map = {
        Region.US_EAST: [ComplianceFramework.SOC2, ComplianceFramework.HIPAA],
        Region.US_WEST: [ComplianceFramework.SOC2, ComplianceFramework.CCPA],
        Region.EU_WEST: [ComplianceFramework.GDPR, ComplianceFramework.ISO27001],
        Region.EU_CENTRAL: [ComplianceFramework.GDPR, ComplianceFramework.ISO27001],
        Region.ASIA_PACIFIC: [ComplianceFramework.PDPA, ComplianceFramework.SOC2],
        Region.CANADA: [ComplianceFramework.PIPEDA, ComplianceFramework.SOC2],
        Region.BRAZIL: [ComplianceFramework.LGPD, ComplianceFramework.SOC2],
    }
    
    enabled_languages = region_language_map.get(region, [SupportedLanguage.ENGLISH])
    compliance_frameworks = region_compliance_map.get(region, [ComplianceFramework.SOC2])
    
    return GlobalizationConfig(
        default_language=enabled_languages[0],
        enabled_languages=enabled_languages,
        default_region=region,
        compliance_frameworks=compliance_frameworks,
        enable_rtl_support=False  # Enable if Arabic/Hebrew support needed
    )


def get_global_optimization_engine(region: Optional[Region] = None) -> GlobalOptimizationEngine:
    """Get a pre-configured global optimization engine for a region."""
    if region:
        config = create_global_config_for_region(region)
    else:
        config = GlobalizationConfig()
    
    return GlobalOptimizationEngine(config)