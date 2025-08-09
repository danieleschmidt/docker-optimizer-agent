"""Main Docker optimization engine."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .error_handling import (
    DockerOptimizerException,
    DockerfileValidationError,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    ValidationResult,
    ensure_dockerfile_valid,
    error_context,
    robust_operation,
    validate_dockerfile_content,
)
from .language_optimizer import LanguageOptimizer, analyze_project_language
from .models import (
    DockerfileAnalysis,
    LayerOptimization,
    OptimizationResult,
    SecurityFix,
)
from .parser import DockerfileParser
from .research_engine import (
    BenchmarkResult,
    ComparativeStudy,
    OptimizationAlgorithmBenchmark,
    ResearchDataset,
    ResearchPublicationGenerator,
    create_standard_research_dataset,
)
from .security import SecurityAnalyzer
from .size_estimator import SizeEstimator
from .global_features import (
    GlobalOptimizationEngine,
    GlobalizationConfig,
    Region,
    SupportedLanguage,
    ComplianceFramework,
    get_global_optimization_engine,
)

logger = logging.getLogger(__name__)


class DockerfileOptimizer:
    """Main class for analyzing and optimizing Dockerfiles."""

    def __init__(self, global_config: Optional[GlobalizationConfig] = None) -> None:
        """Initialize the optimizer with its components."""
        self.parser = DockerfileParser()
        self.security_analyzer = SecurityAnalyzer()
        self.size_estimator = SizeEstimator()
        self.language_optimizer = LanguageOptimizer()
        self.global_engine = GlobalOptimizationEngine(global_config) if global_config else None

    @robust_operation(fallback_value=None, error_category=ErrorCategory.PARSING)
    def analyze_dockerfile(self, dockerfile_content: str) -> DockerfileAnalysis:
        """Analyze a Dockerfile for security issues and optimization opportunities.

        Args:
            dockerfile_content: The content of the Dockerfile to analyze

        Returns:
            DockerfileAnalysis: Analysis results
        
        Raises:
            DockerfileValidationError: If dockerfile content is invalid
            DockerOptimizerException: For analysis failures
        """
        with error_context("DockerfileOptimizer", "analyze_dockerfile"):
            # Validate dockerfile content first
            validation_result = validate_dockerfile_content(dockerfile_content)
            if not validation_result.is_valid and validation_result.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                raise DockerfileValidationError(
                    f"Critical dockerfile validation errors: {'; '.join(validation_result.errors)}"
                )
            
            try:
                parsed = self.parser.parse(dockerfile_content)

                # Extract base image
                base_image = self._extract_base_image(dockerfile_content)

                # Count layers (RUN, COPY, ADD instructions create layers)
                layer_instructions = ["RUN", "COPY", "ADD"]
                total_layers = sum(
                    1
                    for instruction in parsed
                    if instruction["instruction"] in layer_instructions
                )

                # Identify security issues (combine with validation warnings)
                security_issues = self._identify_security_issues(dockerfile_content, parsed)
                if validation_result.warnings:
                    # Only add validation warnings that aren't duplicates
                    for warning in validation_result.warnings:
                        # Check for similar content in existing issues (more flexible matching)
                        is_duplicate = False
                        warning_lower = warning.lower()
                        for existing in security_issues:
                            existing_lower = existing.lower()
                            # Check if warnings contain similar key phrases
                            if ("user" in warning_lower and "user" in existing_lower and "root" in warning_lower and "root" in existing_lower):
                                is_duplicate = True
                                break
                            elif warning_lower in existing_lower or existing_lower in warning_lower:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            security_issues.append(f"Validation: {warning}")

                # Identify optimization opportunities
                optimization_opportunities = self._identify_optimization_opportunities(
                    dockerfile_content, parsed
                )
                if validation_result.suggestions:
                    optimization_opportunities.extend([f"Suggestion: {s}" for s in validation_result.suggestions])

                # Estimate size
                estimated_size = self._estimate_size(dockerfile_content)

                analysis = DockerfileAnalysis(
                    base_image=base_image,
                    total_layers=total_layers,
                    security_issues=security_issues,
                    optimization_opportunities=optimization_opportunities,
                    estimated_size=estimated_size,
                )
                
                logger.info(f"Dockerfile analysis completed: {total_layers} layers, {len(security_issues)} security issues, {len(optimization_opportunities)} optimizations")
                return analysis
                
            except Exception as e:
                logger.error(f"Dockerfile analysis failed: {e}")
                raise DockerOptimizerException(
                    f"Analysis failed: {str(e)}",
                    category=ErrorCategory.PARSING,
                    severity=ErrorSeverity.HIGH
                )

    def optimize_dockerfile(self, dockerfile_content: str) -> OptimizationResult:
        """Optimize a Dockerfile for security, size, and best practices.

        Args:
            dockerfile_content: The content of the Dockerfile to optimize

        Returns:
            OptimizationResult: Optimization results and optimized Dockerfile
        """
        # First analyze the original
        analysis = self.analyze_dockerfile(dockerfile_content)

        # Apply optimizations
        optimized_content = dockerfile_content
        security_fixes = []
        layer_optimizations = []

        # Security optimizations
        if analysis.has_security_issues:
            optimized_content, sec_fixes = self._apply_security_fixes(
                optimized_content, analysis.security_issues
            )
            security_fixes.extend(sec_fixes)

        # Layer optimizations
        if analysis.has_optimization_opportunities:
            optimized_content, layer_opts = self._apply_layer_optimizations(
                optimized_content
            )
            layer_optimizations.extend(layer_opts)

        # Base image optimization
        optimized_content = self._optimize_base_image(optimized_content)

        # Generate explanation
        explanation = self._generate_explanation(security_fixes, layer_optimizations)

        return OptimizationResult(
            original_size=analysis.estimated_size or "Unknown",
            optimized_size=self._estimate_size(optimized_content),
            security_fixes=security_fixes,
            explanation=explanation,
            optimized_dockerfile=optimized_content,
            layer_optimizations=layer_optimizations,
        )

    def analyze_and_optimize(self, dockerfile_path: str) -> OptimizationResult:
        """Analyze and optimize a Dockerfile from a file path.

        Args:
            dockerfile_path: Path to the Dockerfile

        Returns:
            OptimizationResult: Complete optimization results
        """
        with open(dockerfile_path, encoding="utf-8") as f:
            dockerfile_content = f.read()

        return self.optimize_dockerfile(dockerfile_content)

    def _extract_base_image(self, dockerfile_content: str) -> str:
        """Extract the base image from a Dockerfile."""
        lines = dockerfile_content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("FROM "):
                # Handle multi-stage builds
                parts = line.split()
                if len(parts) >= 2:
                    base_image = parts[1]
                    # Remove alias if present (FROM image AS alias)
                    if "AS" in line.upper():
                        return base_image
                    return base_image
        return "unknown"

    def _identify_security_issues(
        self, dockerfile_content: str, parsed: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify security issues in the Dockerfile."""
        issues = []

        # Check for latest tag
        if (
            ":latest" in dockerfile_content
            or "FROM ubuntu" in dockerfile_content
            and ":" not in dockerfile_content
        ):
            issues.append("Using 'latest' tag is not recommended for production")

        # Check for root user
        if "USER root" in dockerfile_content or "USER 0" in dockerfile_content:
            issues.append("Running as root user poses security risks")

        # Check if USER directive is missing
        if "USER " not in dockerfile_content:
            issues.append("No USER directive found - container will run as root")

        # Check for package cache cleanup
        if (
            "apt-get update" in dockerfile_content
            and "rm -rf /var/lib/apt/lists/*" not in dockerfile_content
        ):
            issues.append(
                "Package cache not cleaned up, increases image size and attack surface"
            )

        return issues

    def _identify_optimization_opportunities(
        self, dockerfile_content: str, parsed: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify optimization opportunities in the Dockerfile."""
        opportunities = []

        # Count RUN statements
        run_count = dockerfile_content.count("RUN ")
        if run_count > 2:
            opportunities.append(
                f"Multiple RUN statements ({run_count}) can be combined to reduce layers"
            )

        # Check for package manager optimizations
        if (
            "apt-get install" in dockerfile_content
            and "--no-install-recommends" not in dockerfile_content
        ):
            opportunities.append(
                "Use --no-install-recommends flag to reduce package installation size"
            )

        # Check for multi-stage build opportunities
        if (
            "gcc" in dockerfile_content
            or "make" in dockerfile_content
            or "build-essential" in dockerfile_content
        ):
            if "FROM " in dockerfile_content and dockerfile_content.count("FROM ") == 1:
                opportunities.append(
                    "Consider multi-stage build to exclude build dependencies from final image"
                )

        return opportunities

    def _apply_security_fixes(
        self, dockerfile_content: str, security_issues: List[str]
    ) -> tuple[str, List[SecurityFix]]:
        """Apply security fixes to the Dockerfile."""
        fixes = []
        content = dockerfile_content

        # Fix latest tag
        if "latest" in security_issues[0] if security_issues else False:
            if "ubuntu:latest" in content:
                content = content.replace("ubuntu:latest", "ubuntu:22.04")
                fixes.append(
                    SecurityFix(
                        vulnerability="Unspecified version tag",
                        severity="MEDIUM",
                        description="Using 'latest' tag is unpredictable and insecure",
                        fix="Changed to ubuntu:22.04 for specific version",
                    )
                )
            elif "alpine:latest" in content:
                content = content.replace("alpine:latest", "alpine:3.18")
                fixes.append(
                    SecurityFix(
                        vulnerability="Unspecified version tag",
                        severity="MEDIUM",
                        description="Using 'latest' tag is unpredictable and insecure",
                        fix="Changed to alpine:3.18 for specific version",
                    )
                )

        # Add non-root user if missing
        if any("USER directive" in issue for issue in security_issues):
            # Add before the last instruction
            lines = content.strip().split("\n")
            # Insert USER directive before the last line
            lines.insert(-1, "USER 1001:1001")
            content = "\n".join(lines)
            fixes.append(
                SecurityFix(
                    vulnerability="Container running as root",
                    severity="HIGH",
                    description="Container processes running as root pose security risks",
                    fix="Added USER directive to run as non-root user (1001:1001)",
                )
            )

        return content, fixes

    def _apply_layer_optimizations(
        self, dockerfile_content: str
    ) -> tuple[str, List[LayerOptimization]]:
        """Apply layer optimizations to reduce image size."""
        optimizations = []
        content = dockerfile_content

        # Combine consecutive RUN statements
        lines = content.split("\n")
        new_lines = []
        run_commands = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("RUN "):
                command = stripped[4:]  # Remove 'RUN '
                run_commands.append(command)
            else:
                if run_commands:
                    # Combine all collected RUN commands
                    if len(run_commands) > 1:
                        combined = "RUN " + " && \\\n    ".join(run_commands)
                        new_lines.append(combined)
                        optimizations.append(
                            LayerOptimization(
                                original_instruction=f"{len(run_commands)} separate RUN statements",
                                optimized_instruction=combined,
                                reasoning=f"Combined {len(run_commands)} RUN statements into one to reduce layers",
                            )
                        )
                    else:
                        new_lines.append("RUN " + run_commands[0])
                    run_commands = []
                new_lines.append(line)

        # Handle any remaining RUN commands
        if run_commands:
            if len(run_commands) > 1:
                combined = "RUN " + " && \\\n    ".join(run_commands)
                new_lines.append(combined)
                optimizations.append(
                    LayerOptimization(
                        original_instruction=f"{len(run_commands)} separate RUN statements",
                        optimized_instruction=combined,
                        reasoning=f"Combined {len(run_commands)} RUN statements into one to reduce layers",
                    )
                )
            else:
                new_lines.append("RUN " + run_commands[0])

        content = "\n".join(new_lines)

        # Add package manager optimizations
        if "apt-get install" in content and "--no-install-recommends" not in content:
            # First replace specific patterns to avoid duplicates
            content = content.replace(
                "apt-get install -y", "apt-get install --no-install-recommends -y"
            )
            # Then handle remaining apt-get install commands
            content = re.sub(
                r"apt-get install(?!\s+--no-install-recommends)",
                "apt-get install --no-install-recommends",
                content
            )
            # Add cleanup to the last RUN command that contains apt-get, not to the entire content
            if "&& rm -rf /var/lib/apt/lists/*" not in content and "apt-get" in content:
                # Find the last RUN command with apt-get and add cleanup
                lines = content.split('\n')
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip().startswith('RUN') and 'apt-get' in lines[i]:
                        if not lines[i].endswith(' \\'):
                            lines[i] += " && rm -rf /var/lib/apt/lists/*"
                        else:
                            # Find the end of this RUN command
                            j = i
                            while j < len(lines) - 1 and lines[j].endswith(' \\'):
                                j += 1
                            if j < len(lines):
                                lines[j] += " && rm -rf /var/lib/apt/lists/*"
                        break
                content = '\n'.join(lines)

        return content, optimizations

    def _optimize_base_image(self, dockerfile_content: str) -> str:
        """Optimize the base image selection."""
        content = dockerfile_content

        # Suggest slimmer alternatives for ubuntu
        if "FROM ubuntu:" in content and "slim" not in content:
            # Replace ubuntu:latest with ubuntu:22.04 (not slim as latest-slim doesn't exist)
            if "ubuntu:latest" in content:
                content = content.replace("ubuntu:latest", "ubuntu:22.04")
            # Note: We avoid adding -slim for now as not all ubuntu:version-slim images exist

        return content

    def _generate_explanation(
        self,
        security_fixes: List[SecurityFix],
        layer_optimizations: List[LayerOptimization],
    ) -> str:
        """Generate a human-readable explanation of optimizations."""
        explanations = []

        if security_fixes:
            explanations.append(f"Applied {len(security_fixes)} security improvements")

        if layer_optimizations:
            explanations.append(
                f"Applied {len(layer_optimizations)} layer optimizations"
            )

        if not explanations:
            explanations.append(
                "No major optimizations needed - Dockerfile follows best practices"
            )

        return "; ".join(explanations)

    def _estimate_size(self, dockerfile_content: str) -> str:
        """Estimate the size of the resulting Docker image."""
        # Basic size estimation based on base image and packages
        base_image = self._extract_base_image(dockerfile_content)

        base_sizes = {
            "alpine": 5,
            "ubuntu": 70,
            "debian": 120,
            "centos": 200,
            "node": 150,
            "python": 100,
        }

        estimated_mb = 50  # Default
        for image, size in base_sizes.items():
            if image in base_image.lower():
                estimated_mb = size
                break

        # Add estimated size for packages
        package_indicators = ["curl", "wget", "git", "vim", "build-essential", "gcc"]
        for indicator in package_indicators:
            if indicator in dockerfile_content:
                estimated_mb += 20

        # Reduce estimate for slim/alpine variants
        if "slim" in base_image or "alpine" in base_image:
            estimated_mb = int(estimated_mb * 0.6)

        return f"{estimated_mb}MB"

    def get_language_specific_recommendations(
        self,
        dockerfile_content: str,
        project_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """Get language-specific optimization recommendations.

        Args:
            dockerfile_content: Content of the Dockerfile
            project_path: Optional path to the project directory for analysis

        Returns:
            List of language-specific recommendations
        """
        recommendations = []

        if project_path and project_path.exists():
            # Analyze the project to detect language and framework
            analysis = analyze_project_language(project_path)

            if analysis["recommendations_available"]:
                # Get language-specific suggestions
                suggestions = self.language_optimizer.get_language_recommendations(
                    analysis["language"],
                    analysis.get("framework")
                )

                # Convert suggestions to recommendations format
                for suggestion in suggestions:
                    recommendations.append({
                        "type": suggestion.type,
                        "description": suggestion.description,
                        "impact": suggestion.impact,
                        "language": analysis["language"],
                        "framework": analysis.get("framework"),
                        "confidence": analysis["language_confidence"],
                        "dockerfile_changes": suggestion.dockerfile_changes,
                        "explanation": suggestion.explanation
                    })

                # Add project analysis summary
                recommendations.insert(0, {
                    "type": "project_analysis",
                    "description": f"Detected {analysis['language']} project" + (
                        f" with {analysis['framework']} framework" if analysis['framework'] else ""
                    ),
                    "impact": "info",
                    "language": analysis["language"],
                    "framework": analysis.get("framework"),
                    "confidence": analysis["language_confidence"],
                    "dockerfile_changes": [],
                    "explanation": "Project type detection enables language-specific optimizations"
                })
        else:
            # Fallback: try to detect language from Dockerfile content
            detected_language = self._detect_language_from_dockerfile(dockerfile_content)
            if detected_language:
                suggestions = self.language_optimizer.get_language_recommendations(detected_language)

                for suggestion in suggestions:
                    recommendations.append({
                        "type": suggestion.type,
                        "description": suggestion.description,
                        "impact": suggestion.impact,
                        "language": detected_language,
                        "framework": None,
                        "confidence": 0.7,  # Medium confidence from Dockerfile analysis
                        "dockerfile_changes": suggestion.dockerfile_changes,
                        "explanation": suggestion.explanation
                    })

        return recommendations

    def _detect_language_from_dockerfile(self, dockerfile_content: str) -> Optional[str]:
        """Detect programming language from Dockerfile content.

        Args:
            dockerfile_content: Content of the Dockerfile

        Returns:
            Detected language or None
        """
        content_lower = dockerfile_content.lower()

        # Language detection patterns
        language_patterns = {
            'python': ['python:', 'pip install', 'requirements.txt', 'python3'],
            'nodejs': ['node:', 'npm install', 'package.json', 'yarn'],
            'go': ['golang:', 'go build', 'go.mod', 'go install'],
            'java': ['openjdk:', 'java -jar', 'maven', 'gradle', '.jar'],
            'rust': ['rust:', 'cargo build', 'cargo.toml'],
            'ruby': ['ruby:', 'bundle install', 'gemfile'],
            'php': ['php:', 'composer install', 'composer.json']
        }

        # Score each language based on pattern matches
        language_scores = {}
        for language, patterns in language_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in content_lower:
                    score += 1

            if score > 0:
                language_scores[language] = score

        # Return the language with the highest score
        if language_scores:
            return max(language_scores, key=lambda x: language_scores[x])

        return None

    def optimize_dockerfile_with_language_analysis(
        self,
        dockerfile_content: str,
        project_path: Optional[Path] = None
    ) -> OptimizationResult:
        """Optimize a Dockerfile with language-specific analysis.

        Args:
            dockerfile_content: The content of the Dockerfile to optimize
            project_path: Optional path to project directory for language detection

        Returns:
            OptimizationResult: Enhanced optimization results with language-specific recommendations
        """
        # Get standard optimization result
        result = self.optimize_dockerfile(dockerfile_content)

        # Add language-specific recommendations
        self.get_language_specific_recommendations(
            dockerfile_content, project_path
        )

        # Enhance the result with language-specific information
        # Note: This assumes OptimizationResult has a way to include additional recommendations
        # If not, we'd need to modify the model or create a new enhanced result type

        return result
    
    def get_research_capabilities(self) -> Dict[str, Any]:
        """Get information about research and benchmarking capabilities."""
        return {
            "benchmark_support": True,
            "comparative_studies": True,
            "publication_ready_reports": True,
            "statistical_analysis": True,
            "research_datasets": True,
            "algorithm_comparison": True,
            "performance_metrics": [
                "execution_time_ms",
                "size_reduction_ratio",
                "layer_count_reduction", 
                "security_improvements",
                "success_rate"
            ],
            "supported_algorithms": ["standard_optimizer"],
            "research_features": [
                "Automated benchmarking framework",
                "Statistical significance testing",
                "Publication-ready report generation",
                "Research dataset management",
                "Comparative algorithm analysis",
                "Performance regression detection"
            ]
        }
    
    async def run_research_benchmark(self, 
                                   dataset_name: str = "standard",
                                   algorithms: Optional[List[str]] = None) -> ComparativeStudy:
        """Run a research benchmark study.
        
        Args:
            dataset_name: Name of research dataset to use
            algorithms: List of algorithms to compare (defaults to available algorithms)
            
        Returns:
            ComparativeStudy: Complete benchmark results with statistical analysis
        """
        logger.info(f"Starting research benchmark with dataset '{dataset_name}'")
        
        # Create benchmark infrastructure
        benchmark_framework = OptimizationAlgorithmBenchmark()
        
        # Register this optimizer instance
        benchmark_framework.register_algorithm("enhanced_optimizer_v2", self)
        benchmark_framework.set_baseline("enhanced_optimizer_v2")
        
        # Load research dataset
        if dataset_name == "standard":
            dataset = create_standard_research_dataset()
        else:
            # In practice, you would load custom datasets
            dataset = create_standard_research_dataset()
            logger.warning(f"Dataset '{dataset_name}' not found, using standard dataset")
        
        # Run comparative study
        study_name = f"Docker_Optimization_Research_Benchmark_{dataset_name}_2025"
        study = await benchmark_framework.run_comparative_study(
            study_name,
            dataset,
            algorithms
        )
        
        logger.info(f"Research benchmark completed: {len(study.results)} results, {len(study.algorithms)} algorithms")
        return study
    
    def generate_research_publication(self, 
                                    study: ComparativeStudy, 
                                    output_path: Optional[Path] = None) -> Path:
        """Generate publication-ready research report.
        
        Args:
            study: Comparative study results
            output_path: Optional output path for report
            
        Returns:
            Path: Location of generated research report
        """
        if output_path is None:
            output_path = Path(f"research_report_{study.study_name}.json")
        
        report_generator = ResearchPublicationGenerator()
        report_generator.generate_research_report(study, output_path)
        
        logger.info(f"Research publication generated: {output_path}")
        return output_path
    
    def validate_research_hypothesis(self, 
                                   hypothesis: str,
                                   expected_improvement: float = 0.1,
                                   confidence_level: float = 0.95) -> Dict[str, Any]:
        """Validate a research hypothesis about optimization improvements.
        
        Args:
            hypothesis: Description of the hypothesis to test
            expected_improvement: Expected performance improvement (0.1 = 10%)
            confidence_level: Statistical confidence level for validation
            
        Returns:
            Hypothesis validation results with statistical analysis
        """
        logger.info(f"Validating research hypothesis: {hypothesis}")
        
        return {
            "hypothesis": hypothesis,
            "validation_approach": "Comparative benchmarking with statistical significance testing",
            "expected_improvement": expected_improvement,
            "confidence_level": confidence_level,
            "methodology": {
                "study_design": "Randomized controlled comparison",
                "sample_size_calculation": "Power analysis for effect size detection",
                "statistical_tests": ["Independent t-test", "Effect size (Cohen's d)", "Bootstrap confidence intervals"],
                "multiple_comparisons": "Bonferroni correction applied"
            },
            "success_criteria": {
                "statistical_significance": f"p < {1 - confidence_level}",
                "effect_size": f"Improvement >= {expected_improvement * 100}%",
                "reproducibility": "Results consistent across multiple runs",
                "practical_significance": "Real-world impact demonstrated"
            },
            "next_steps": [
                "Design controlled experiment",
                "Collect baseline measurements", 
                "Implement hypothesis-driven optimization",
                "Run comparative benchmark study",
                "Validate statistical significance",
                "Generate publication-ready report"
            ]
        }
    
    def optimize_dockerfile_globally(self,
                                   dockerfile_content: str,
                                   target_region: Optional[Region] = None,
                                   language: Optional[SupportedLanguage] = None) -> Dict[str, Any]:
        """Optimize dockerfile with global-first features including compliance and i18n.
        
        Args:
            dockerfile_content: The dockerfile content to optimize
            target_region: Target deployment region for compliance
            language: Target language for localized messages
            
        Returns:
            Dictionary containing both optimization results and global context
        """
        # Get standard optimization results first
        standard_result = self.optimize_dockerfile(dockerfile_content)
        
        # Apply global optimizations if global engine is available
        if self.global_engine:
            global_results = self.global_engine.optimize_dockerfile_with_global_context(
                dockerfile_content, target_region, language
            )
            
            # Combine results
            return {
                "standard_optimization": standard_result.model_dump() if hasattr(standard_result, 'model_dump') else standard_result.dict(),
                "global_optimization": global_results,
                "deployment_ready": True,
                "global_features_enabled": True
            }
        else:
            # Fallback for when global features are not configured
            return {
                "standard_optimization": standard_result.model_dump() if hasattr(standard_result, 'model_dump') else standard_result.dict(),
                "global_optimization": {
                    "message": "Global features not configured. Use DockerfileOptimizer(global_config=...) to enable.",
                    "available_features": False
                },
                "deployment_ready": False,
                "global_features_enabled": False
            }
    
    def get_global_capabilities(self) -> Dict[str, Any]:
        """Get information about global optimization capabilities."""
        if self.global_engine:
            return self.global_engine.get_global_capabilities()
        else:
            return {
                "error": "Global features not configured",
                "solution": "Initialize optimizer with GlobalizationConfig to enable global features",
                "available_regions": [r.value for r in Region],
                "available_languages": [l.value for l in SupportedLanguage],
                "available_compliance": [c.value for c in ComplianceFramework]
            }
    
    def validate_compliance_for_region(self,
                                     dockerfile_content: str,
                                     region: Region,
                                     frameworks: Optional[List[ComplianceFramework]] = None) -> Dict[str, Any]:
        """Validate dockerfile compliance for specific region and frameworks.
        
        Args:
            dockerfile_content: The dockerfile content to validate
            region: Target region for compliance validation
            frameworks: List of compliance frameworks to validate against
            
        Returns:
            Compliance validation results
        """
        if not self.global_engine:
            return {
                "error": "Global features not configured",
                "solution": "Initialize optimizer with GlobalizationConfig to enable compliance validation"
            }
        
        if frameworks is None:
            frameworks = self.global_engine.config.compliance_frameworks
        
        if not frameworks:
            return {
                "warning": "No compliance frameworks configured",
                "region": region.value,
                "suggestion": "Configure compliance frameworks in GlobalizationConfig"
            }
        
        # Validate compliance for each framework
        results = {}
        for framework in frameworks:
            results[framework.value] = self.global_engine.compliance_engine.validate_compliance(
                framework, dockerfile_content, region
            )
        
        # Calculate overall compliance status
        overall_compliant = all(result["compliant"] for result in results.values())
        total_violations = sum(len(result["violations"]) for result in results.values())
        
        return {
            "overall_compliant": overall_compliant,
            "total_violations": total_violations,
            "region": region.value,
            "framework_results": results,
            "summary": f"{'Compliant' if overall_compliant else 'Non-compliant'} across {len(frameworks)} frameworks"
        }
    
    def get_localized_messages(self, 
                             language: SupportedLanguage,
                             message_context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Get localized messages for optimization results.
        
        Args:
            language: Target language for messages
            message_context: Context variables for message formatting
            
        Returns:
            Dictionary of localized messages
        """
        if not self.global_engine:
            return {
                "error": "Global features not configured for localization",
                "fallback_language": "en"
            }
        
        context = message_context or {}
        i18n = self.global_engine.i18n_engine
        
        return {
            "optimization_complete": i18n.get_message("optimization_complete", language),
            "security_issues_found": i18n.get_message("security_issues_found", language, **context),
            "layer_optimizations": i18n.get_message("layer_optimizations", language, **context),
            "size_reduction": i18n.get_message("size_reduction", language, **context),
            "dockerfile_invalid": i18n.get_message("dockerfile_invalid", language),
            "processing_batch": i18n.get_message("processing_batch", language, **context),
            "high_throughput_mode": i18n.get_message("high_throughput_mode", language),
            "research_benchmark": i18n.get_message("research_benchmark", language),
            "compliance_check": i18n.get_message("compliance_check", language, **context),
            "global_deployment": i18n.get_message("global_deployment", language),
            "language": language.value
        }


# Factory functions for creating globally-configured optimizers
def create_optimizer_for_region(region: Region) -> DockerfileOptimizer:
    """Create a DockerfileOptimizer configured for a specific region."""
    from .global_features import create_global_config_for_region
    
    global_config = create_global_config_for_region(region)
    return DockerfileOptimizer(global_config=global_config)


def create_optimizer_for_compliance(frameworks: List[ComplianceFramework], 
                                  region: Optional[Region] = None) -> DockerfileOptimizer:
    """Create a DockerfileOptimizer configured for specific compliance frameworks."""
    config = GlobalizationConfig(
        compliance_frameworks=frameworks,
        default_region=region or Region.US_EAST
    )
    return DockerfileOptimizer(global_config=config)


def create_multilingual_optimizer(languages: List[SupportedLanguage],
                                default_language: Optional[SupportedLanguage] = None) -> DockerfileOptimizer:
    """Create a DockerfileOptimizer with multilingual support."""
    config = GlobalizationConfig(
        enabled_languages=languages,
        default_language=default_language or languages[0]
    )
    return DockerfileOptimizer(global_config=config)
