"""Language-specific optimization patterns for Docker Optimizer Agent.

This module provides intelligent optimization patterns based on:
1. Project type detection (Python, Node.js, Go, Java, Rust, etc.)
2. Language-specific base image recommendations
3. Framework-aware optimizations (Django, Express, Spring, etc.)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .models import OptimizationSuggestion


class ProjectTypeDetector:
    """Detects project type and framework from project files."""

    def __init__(self, project_path: Path) -> None:
        """Initialize the project type detector.

        Args:
            project_path: Path to the project directory
        """
        self.project_path = project_path
        self.detected_files: Dict[str, bool] = {}
        self._scan_project_files()

    def _scan_project_files(self) -> None:
        """Scan project directory for language and framework indicators."""
        # Define file patterns for detection
        detection_patterns = {
            # Python
            'requirements.txt': 'python_pip',
            'pyproject.toml': 'python_modern',
            'setup.py': 'python_setuptools',
            'Pipfile': 'python_pipenv',
            'poetry.lock': 'python_poetry',
            'manage.py': 'python_django',
            'app.py': 'python_flask',
            'main.py': 'python_generic',

            # Node.js
            'package.json': 'nodejs_npm',
            'yarn.lock': 'nodejs_yarn',
            'pnpm-lock.yaml': 'nodejs_pnpm',
            'next.config.js': 'nodejs_nextjs',
            'nuxt.config.js': 'nodejs_nuxtjs',
            'angular.json': 'nodejs_angular',
            'vue.config.js': 'nodejs_vue',
            'server.js': 'nodejs_express',

            # Go
            'go.mod': 'go_modules',
            'go.sum': 'go_modules',
            'main.go': 'go_generic',
            'Gopkg.toml': 'go_dep',

            # Java
            'pom.xml': 'java_maven',
            'build.gradle': 'java_gradle',
            'build.gradle.kts': 'java_gradle_kotlin',
            'application.properties': 'java_spring',
            'application.yml': 'java_spring',

            # Rust
            'Cargo.toml': 'rust_cargo',
            'Cargo.lock': 'rust_cargo',

            # Ruby
            'Gemfile': 'ruby_bundler',
            'config/application.rb': 'ruby_rails',

            # PHP
            'composer.json': 'php_composer',
            'artisan': 'php_laravel',

            # .NET
            '*.csproj': 'dotnet_csproj',
            '*.sln': 'dotnet_solution',

            # Frontend
            'webpack.config.js': 'frontend_webpack',
            'vite.config.js': 'frontend_vite',
            'rollup.config.js': 'frontend_rollup',
        }

        # Scan for files
        for pattern, file_type in detection_patterns.items():
            if '*' in pattern:
                # Handle glob patterns
                matches = list(self.project_path.glob(pattern))
                self.detected_files[file_type] = len(matches) > 0
            else:
                # Handle exact file names
                file_path = self.project_path / pattern
                self.detected_files[file_type] = file_path.exists()

    def detect_primary_language(self) -> Tuple[str, float]:
        """Detect the primary programming language.

        Returns:
            Tuple of (language, confidence_score)
        """
        language_scores = {
            'python': 0.0,
            'nodejs': 0.0,
            'go': 0.0,
            'java': 0.0,
            'rust': 0.0,
            'ruby': 0.0,
            'php': 0.0,
            'dotnet': 0.0,
        }

        # Score based on detected files
        file_weights = {
            # Python
            'python_modern': 0.9,      # pyproject.toml is modern Python
            'python_django': 0.8,      # manage.py indicates Django
            'python_flask': 0.7,       # app.py could be Flask
            'python_pip': 0.6,         # requirements.txt is common
            'python_setuptools': 0.6,  # setup.py is traditional
            'python_pipenv': 0.7,      # Pipfile indicates Pipenv
            'python_poetry': 0.8,      # poetry.lock indicates Poetry
            'python_generic': 0.4,     # main.py could be anything

            # Node.js
            'nodejs_npm': 0.8,         # package.json is definitive
            'nodejs_yarn': 0.9,        # yarn.lock indicates Yarn
            'nodejs_pnpm': 0.9,        # pnpm-lock.yaml indicates pnpm
            'nodejs_nextjs': 0.9,      # next.config.js indicates Next.js
            'nodejs_nuxtjs': 0.9,      # nuxt.config.js indicates Nuxt.js
            'nodejs_angular': 0.9,     # angular.json indicates Angular
            'nodejs_vue': 0.8,         # vue.config.js indicates Vue
            'nodejs_express': 0.7,     # server.js could be Express

            # Go
            'go_modules': 0.9,         # go.mod is definitive
            'go_generic': 0.6,         # main.go could be anything
            'go_dep': 0.7,             # Gopkg.toml is older dependency management

            # Java
            'java_maven': 0.9,         # pom.xml indicates Maven
            'java_gradle': 0.9,        # build.gradle indicates Gradle
            'java_gradle_kotlin': 0.9, # Gradle with Kotlin DSL
            'java_spring': 0.8,        # Spring configuration files

            # Rust
            'rust_cargo': 0.9,         # Cargo.toml is definitive

            # Ruby
            'ruby_bundler': 0.8,       # Gemfile indicates Ruby
            'ruby_rails': 0.9,         # Rails-specific file

            # PHP
            'php_composer': 0.8,       # composer.json indicates PHP
            'php_laravel': 0.9,        # artisan indicates Laravel

            # .NET
            'dotnet_csproj': 0.9,      # .csproj indicates .NET
            'dotnet_solution': 0.8,    # .sln indicates .NET solution
        }

        # Calculate scores
        for file_type, detected in self.detected_files.items():
            if detected and file_type in file_weights:
                weight = file_weights[file_type]

                # Map file types to languages
                if file_type.startswith('python_'):
                    language_scores['python'] += weight
                elif file_type.startswith('nodejs_'):
                    language_scores['nodejs'] += weight
                elif file_type.startswith('go_'):
                    language_scores['go'] += weight
                elif file_type.startswith('java_'):
                    language_scores['java'] += weight
                elif file_type.startswith('rust_'):
                    language_scores['rust'] += weight
                elif file_type.startswith('ruby_'):
                    language_scores['ruby'] += weight
                elif file_type.startswith('php_'):
                    language_scores['php'] += weight
                elif file_type.startswith('dotnet_'):
                    language_scores['dotnet'] += weight

        # Find the highest scoring language
        if not any(language_scores.values()):
            return 'unknown', 0.0

        best_language = max(language_scores, key=lambda x: language_scores[x])
        confidence = min(language_scores[best_language], 1.0)

        return best_language, confidence

    def detect_framework(self) -> Tuple[Optional[str], float]:
        """Detect the primary framework being used.

        Returns:
            Tuple of (framework, confidence_score)
        """
        framework_indicators = {
            # Python frameworks
            'django': ['python_django', 'python_pip'],
            'flask': ['python_flask', 'python_pip'],
            'fastapi': ['python_modern'],  # Would need content analysis

            # Node.js frameworks
            'nextjs': ['nodejs_nextjs', 'nodejs_npm'],
            'nuxtjs': ['nodejs_nuxtjs', 'nodejs_npm'],
            'angular': ['nodejs_angular', 'nodejs_npm'],
            'vue': ['nodejs_vue', 'nodejs_npm'],
            'express': ['nodejs_express', 'nodejs_npm'],
            'react': ['nodejs_npm'],  # Would need content analysis

            # Java frameworks
            'spring': ['java_spring', 'java_maven'],
            'spring_gradle': ['java_spring', 'java_gradle'],

            # Ruby frameworks
            'rails': ['ruby_rails', 'ruby_bundler'],

            # PHP frameworks
            'laravel': ['php_laravel', 'php_composer'],
        }

        framework_scores = {}

        for framework, indicators in framework_indicators.items():
            score = 0.0
            for indicator in indicators:
                if self.detected_files.get(indicator, False):
                    score += 0.5  # Each indicator adds to confidence

            if score > 0:
                framework_scores[framework] = min(score, 1.0)

        if not framework_scores:
            return None, 0.0

        best_framework = max(framework_scores, key=lambda x: framework_scores[x])
        confidence = framework_scores[best_framework]

        return best_framework, confidence


class LanguageOptimizer:
    """Provides language-specific optimization recommendations."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the language optimizer.

        Args:
            config: Optional configuration instance
        """
        self.config = config or Config()
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self._load_optimization_patterns()

    def _load_optimization_patterns(self) -> None:
        """Load language-specific optimization patterns."""
        self.patterns = {
            'python': {
                'base_images': {
                    'production': ['python:3.11-slim', 'python:3.10-slim'],
                    'development': ['python:3.11', 'python:3.10'],
                    'alpine': ['python:3.11-alpine', 'python:3.10-alpine'],
                },
                'multi_stage': True,
                'package_manager': 'pip',
                'common_optimizations': [
                    'Use .dockerignore to exclude __pycache__ and .pyc files',
                    'Install dependencies before copying source code',
                    'Use --no-cache-dir for pip installs',
                    'Consider using wheel packages for faster installs',
                    'Use multi-stage builds to separate build and runtime',
                ],
                'security_recommendations': [
                    'Run as non-root user',
                    'Scan for known vulnerabilities in dependencies',
                    'Use specific package versions in requirements.txt',
                ],
            },
            'nodejs': {
                'base_images': {
                    'production': ['node:18-slim', 'node:16-slim'],
                    'development': ['node:18', 'node:16'],
                    'alpine': ['node:18-alpine', 'node:16-alpine'],
                },
                'multi_stage': True,
                'package_manager': 'npm',
                'common_optimizations': [
                    'Use .dockerignore to exclude node_modules',
                    'Copy package.json and package-lock.json first',
                    'Use npm ci for production builds',
                    'Consider using npm prune --production',
                    'Use multi-stage builds to exclude devDependencies',
                ],
                'security_recommendations': [
                    'Run as non-root user (use USER node)',
                    'Audit dependencies with npm audit',
                    'Use exact versions in package-lock.json',
                ],
            },
            'go': {
                'base_images': {
                    'production': ['golang:1.20-alpine', 'scratch'],
                    'development': ['golang:1.20', 'golang:1.19'],
                    'alpine': ['golang:1.20-alpine', 'golang:1.19-alpine'],
                },
                'multi_stage': True,
                'package_manager': 'go_modules',
                'common_optimizations': [
                    'Use multi-stage builds (build in golang, run in scratch/alpine)',
                    'Use go mod download for dependency caching',
                    'Build static binaries with CGO_ENABLED=0',
                    'Use GOOS and GOARCH for cross-compilation',
                    'Minimize final image size with scratch base',
                ],
                'security_recommendations': [
                    'Use scratch or distroless for minimal attack surface',
                    'Scan dependencies with go list -m all',
                    'Use go mod verify for integrity checks',
                ],
            },
            'java': {
                'base_images': {
                    'production': ['openjdk:17-jre-slim', 'openjdk:11-jre-slim'],
                    'development': ['openjdk:17-jdk', 'openjdk:11-jdk'],
                    'alpine': ['openjdk:17-jre-alpine', 'openjdk:11-jre-alpine'],
                },
                'multi_stage': True,
                'package_manager': 'maven_or_gradle',
                'common_optimizations': [
                    'Use JRE image for runtime instead of JDK',
                    'Use multi-stage builds to separate build and runtime',
                    'Copy dependencies before source for better caching',
                    'Use Maven/Gradle wrapper for consistent builds',
                    'Consider using Maven daemon for faster builds',
                ],
                'security_recommendations': [
                    'Run as non-root user',
                    'Use OWASP dependency check',
                    'Keep JVM updated for security patches',
                ],
            },
            'rust': {
                'base_images': {
                    'production': ['rust:1.70-slim', 'scratch'],
                    'development': ['rust:1.70', 'rust:1.69'],
                    'alpine': ['rust:1.70-alpine', 'rust:1.69-alpine'],
                },
                'multi_stage': True,
                'package_manager': 'cargo',
                'common_optimizations': [
                    'Use multi-stage builds (build in rust, run in scratch/alpine)',
                    'Use cargo chef for dependency caching',
                    'Build release binaries with --release flag',
                    'Strip debug symbols for smaller binaries',
                    'Use musl target for static linking',
                ],
                'security_recommendations': [
                    'Use cargo audit for vulnerability scanning',
                    'Use minimal base images like scratch or distroless',
                    'Keep Rust toolchain updated',
                ],
            },
        }

    def get_language_recommendations(
        self,
        language: str,
        framework: Optional[str] = None,
        optimization_profile: Optional[str] = None,
        profile: Optional[str] = None
    ) -> List[OptimizationSuggestion]:
        """Get optimization recommendations for a specific language.

        Args:
            language: Primary programming language
            framework: Optional framework being used
            optimization_profile: Optimization profile (production, development, etc.)

        Returns:
            List of optimization suggestions
        """
        # Handle both parameter names for backward compatibility
        final_profile = optimization_profile or profile or 'production'

        suggestions = []

        if language not in self.patterns:
            # Generic suggestions for unknown languages
            suggestions.append(OptimizationSuggestion(
                line_number=0,
                suggestion_type="generic",
                priority="MEDIUM",
                message="Use multi-stage builds to reduce image size",
                explanation="Multi-stage builds help separate build dependencies from runtime",
                fix_example="# Use multi-stage builds\n# Copy only necessary files"
            ))
            return suggestions

        pattern = self.patterns[language]

        # Base image recommendations
        base_images = pattern['base_images'].get(final_profile,
                                                pattern['base_images']['production'])
        suggestions.append(OptimizationSuggestion(
            line_number=1,
            suggestion_type="base_image",
            priority="HIGH",
            message=f"Use optimized {language} base image: {base_images[0]}",
            explanation=f"Optimized base image for {language} {final_profile} workloads",
            fix_example=f"FROM {base_images[0]}"
        ))

        # Multi-stage build recommendation
        if pattern.get('multi_stage', False):
            suggestions.append(OptimizationSuggestion(
                line_number=2,
                suggestion_type="build_optimization",
                priority="HIGH",
                message=f"Implement multi-stage build for {language}",
                explanation="Separate build and runtime environments for smaller final image",
                fix_example=f"# Build stage\nFROM {pattern['base_images']['development'][0]} AS builder\n# ... build steps ...\n# Runtime stage\nFROM {base_images[0]}\n# Copy built artifacts from builder stage"
            ))

        # Language-specific optimizations
        for optimization in pattern.get('common_optimizations', []):
            suggestions.append(OptimizationSuggestion(
                line_number=3,
                suggestion_type="optimization",
                priority="MEDIUM",
                message=optimization,
                explanation=f"{language}-specific optimization",
                fix_example=f"# {optimization}"
            ))

        # Security recommendations
        for security_rec in pattern.get('security_recommendations', []):
            suggestions.append(OptimizationSuggestion(
                line_number=4,
                suggestion_type="security",
                priority="HIGH",
                message=security_rec,
                explanation=f"Security best practice for {language}",
                fix_example=f"# {security_rec}"
            ))

        # Framework-specific recommendations
        if framework:
            framework_suggestions = self._get_framework_recommendations(language, framework)
            suggestions.extend(framework_suggestions)

        return suggestions

    def _get_framework_recommendations(
        self,
        language: str,
        framework: str
    ) -> List[OptimizationSuggestion]:
        """Get framework-specific optimization recommendations.

        Args:
            language: Programming language
            framework: Framework being used

        Returns:
            List of framework-specific suggestions
        """
        suggestions = []

        # Framework-specific patterns
        framework_patterns = {
            'django': [
                OptimizationSuggestion(
                    line_number=5,
                    suggestion_type="framework_optimization",
                    priority="MEDIUM",
                    message="Use collectstatic for Django static files in build stage",
                    explanation="Pre-collect static files for better Django performance",
                    fix_example="RUN python manage.py collectstatic --noinput\n# Serve static files efficiently"
                ),
                OptimizationSuggestion(
                    line_number=6,
                    suggestion_type="framework_optimization",
                    priority="HIGH",
                    message="Use gunicorn for production Django deployment",
                    explanation="Gunicorn is a production-ready WSGI server for Django",
                    fix_example="RUN pip install gunicorn\nCMD [\"gunicorn\", \"--bind\", \"0.0.0.0:8000\", \"myproject.wsgi\"]"
                )
            ],
            'flask': [
                OptimizationSuggestion(
                    line_number=7,
                    suggestion_type="framework_optimization",
                    priority="HIGH",
                    message="Use gunicorn for production Flask deployment",
                    explanation="Gunicorn provides better performance than Flask dev server",
                    fix_example="RUN pip install gunicorn\nCMD [\"gunicorn\", \"--bind\", \"0.0.0.0:5000\", \"app:app\"]"
                )
            ],
            'express': [
                OptimizationSuggestion(
                    line_number=8,
                    suggestion_type="framework_optimization",
                    priority="MEDIUM",
                    message="Use PM2 for production Express.js deployment",
                    explanation="PM2 provides process management and monitoring for Node.js",
                    fix_example="RUN npm install -g pm2\nCMD [\"pm2-runtime\", \"start\", \"server.js\"]"
                )
            ],
            'nextjs': [
                OptimizationSuggestion(
                    line_number=9,
                    suggestion_type="framework_optimization",
                    priority="HIGH",
                    message="Use Next.js standalone output for smaller images",
                    explanation="Standalone output reduces Next.js bundle size significantly",
                    fix_example="# next.config.js: output: 'standalone'\nCOPY --from=builder /app/.next/standalone ./\nCOPY --from=builder /app/.next/static ./.next/static"
                )
            ],
            'spring': [
                OptimizationSuggestion(
                    line_number=10,
                    suggestion_type="framework_optimization",
                    priority="HIGH",
                    message="Use Spring Boot layered JARs for better caching",
                    explanation="Layered JARs improve Docker layer caching for Spring Boot",
                    fix_example="RUN java -Djarmode=layertools -jar app.jar extract\nCOPY --from=builder dependencies/ ./\nCOPY --from=builder spring-boot-loader/ ./\nCOPY --from=builder snapshot-dependencies/ ./\nCOPY --from=builder application/ ./"
                )
            ]
        }

        if framework in framework_patterns:
            suggestions.extend(framework_patterns[framework])

        return suggestions

    def optimize_dockerfile_for_language(
        self,
        project_path: Path,
        dockerfile_content: str,
        detected_language: str,
        detected_framework: Optional[str] = None
    ) -> Tuple[str, List[OptimizationSuggestion]]:
        """Optimize a Dockerfile based on detected language and framework.

        Args:
            project_path: Path to the project directory
            dockerfile_content: Current Dockerfile content
            detected_language: Detected programming language
            detected_framework: Detected framework (if any)

        Returns:
            Tuple of (optimized_dockerfile, suggestions_applied)
        """
        suggestions = self.get_language_recommendations(
            detected_language,
            detected_framework
        )

        # Apply high-impact suggestions to the Dockerfile
        optimized_content = dockerfile_content
        applied_suggestions = []

        for suggestion in suggestions:
            if suggestion.impact == "high" and suggestion.type in ["base_image", "build_optimization"]:
                # Apply the suggestion (simplified implementation)
                if suggestion.type == "base_image" and "FROM" in optimized_content:
                    # Replace the base image
                    lines = optimized_content.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith('FROM'):
                            if suggestion.dockerfile_changes:
                                lines[i] = suggestion.dockerfile_changes[0]
                                applied_suggestions.append(suggestion)
                                break
                    optimized_content = '\n'.join(lines)

        return optimized_content, applied_suggestions


def analyze_project_language(project_path: Path) -> Dict[str, Any]:
    """Analyze a project to detect language and framework.

    Args:
        project_path: Path to the project directory

    Returns:
        Analysis results including language, framework, and confidence scores
    """
    detector = ProjectTypeDetector(project_path)

    language, lang_confidence = detector.detect_primary_language()
    framework, framework_confidence = detector.detect_framework()

    return {
        'language': language,
        'language_confidence': lang_confidence,
        'framework': framework,
        'framework_confidence': framework_confidence,
        'detected_files': detector.detected_files,
        'recommendations_available': language in ['python', 'nodejs', 'go', 'java', 'rust']
    }
