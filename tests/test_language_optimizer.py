"""Tests for language-specific optimization patterns."""

import tempfile
from pathlib import Path

from docker_optimizer.language_optimizer import (
    LanguageOptimizer,
    ProjectTypeDetector,
    analyze_project_language,
)


class TestProjectTypeDetector:
    """Test project type detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def _create_file(self, filename: str, content: str = ""):
        """Create a test file in the project directory."""
        file_path = self.project_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path

    def test_python_project_detection(self):
        """Test detection of Python projects."""
        self._create_file("requirements.txt", "django>=4.0\npsycopg2>=2.8")
        self._create_file("manage.py", "#!/usr/bin/env python")

        detector = ProjectTypeDetector(self.project_path)
        language, confidence = detector.detect_primary_language()

        assert language == "python"
        assert confidence > 0.8

    def test_python_modern_project_detection(self):
        """Test detection of modern Python projects with pyproject.toml."""
        self._create_file("pyproject.toml", "[tool.poetry]\nname = 'test'")
        self._create_file("poetry.lock", "# Poetry lock file")

        detector = ProjectTypeDetector(self.project_path)
        language, confidence = detector.detect_primary_language()

        assert language == "python"
        assert confidence > 0.8

    def test_nodejs_project_detection(self):
        """Test detection of Node.js projects."""
        self._create_file("package.json", '{"name": "test", "version": "1.0.0"}')
        self._create_file("yarn.lock", "# Yarn lock file")

        detector = ProjectTypeDetector(self.project_path)
        language, confidence = detector.detect_primary_language()

        assert language == "nodejs"
        assert confidence > 0.8

    def test_go_project_detection(self):
        """Test detection of Go projects."""
        self._create_file("go.mod", "module test\ngo 1.20")
        self._create_file("main.go", "package main\nfunc main() {}")

        detector = ProjectTypeDetector(self.project_path)
        language, confidence = detector.detect_primary_language()

        assert language == "go"
        assert confidence > 0.8

    def test_java_project_detection(self):
        """Test detection of Java projects."""
        self._create_file("pom.xml", "<project><groupId>test</groupId></project>")
        self._create_file("application.properties", "server.port=8080")

        detector = ProjectTypeDetector(self.project_path)
        language, confidence = detector.detect_primary_language()

        assert language == "java"
        assert confidence > 0.8

    def test_rust_project_detection(self):
        """Test detection of Rust projects."""
        self._create_file("Cargo.toml", "[package]\nname = 'test'")
        self._create_file("Cargo.lock", "# Cargo lock file")

        detector = ProjectTypeDetector(self.project_path)
        language, confidence = detector.detect_primary_language()

        assert language == "rust"
        assert confidence > 0.8

    def test_django_framework_detection(self):
        """Test detection of Django framework."""
        self._create_file("requirements.txt", "django>=4.0")
        self._create_file("manage.py", "#!/usr/bin/env python")

        detector = ProjectTypeDetector(self.project_path)
        framework, confidence = detector.detect_framework()

        assert framework == "django"
        assert confidence > 0.5

    def test_nextjs_framework_detection(self):
        """Test detection of Next.js framework."""
        self._create_file("package.json", '{"name": "test"}')
        self._create_file("next.config.js", "module.exports = {}")

        detector = ProjectTypeDetector(self.project_path)
        framework, confidence = detector.detect_framework()

        assert framework == "nextjs"
        assert confidence > 0.5

    def test_spring_framework_detection(self):
        """Test detection of Spring framework."""
        self._create_file("pom.xml", "<project><groupId>test</groupId></project>")
        self._create_file("application.yml", "server:\n  port: 8080")

        detector = ProjectTypeDetector(self.project_path)
        framework, confidence = detector.detect_framework()

        assert framework == "spring"
        assert confidence > 0.5

    def test_unknown_project_detection(self):
        """Test handling of unknown project types."""
        self._create_file("README.md", "# Test project")

        detector = ProjectTypeDetector(self.project_path)
        language, confidence = detector.detect_primary_language()

        assert language == "unknown"
        assert confidence == 0.0

    def test_mixed_project_detection(self):
        """Test handling of projects with multiple languages."""
        # Create both Python and Node.js files
        self._create_file("requirements.txt", "django>=4.0")
        self._create_file("package.json", '{"name": "test"}')

        detector = ProjectTypeDetector(self.project_path)
        language, confidence = detector.detect_primary_language()

        # Should detect the language with higher confidence
        assert language in ["python", "nodejs"]
        assert confidence > 0.5


class TestLanguageOptimizer:
    """Test language-specific optimization recommendations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = LanguageOptimizer()

    def test_python_recommendations(self):
        """Test Python-specific optimization recommendations."""
        suggestions = self.optimizer.get_language_recommendations("python")

        assert len(suggestions) > 0

        # Check for base image suggestion
        base_image_suggestions = [s for s in suggestions if s.type == "base_image"]
        assert len(base_image_suggestions) > 0
        assert "python:" in base_image_suggestions[0].description

        # Check for multi-stage build suggestion
        build_suggestions = [s for s in suggestions if s.type == "build_optimization"]
        assert len(build_suggestions) > 0

        # Check for Python-specific optimizations
        optimization_texts = [s.description for s in suggestions]
        assert any("pip" in text for text in optimization_texts)
        assert any("requirements.txt" in text or "dependencies" in text for text in optimization_texts)

    def test_nodejs_recommendations(self):
        """Test Node.js-specific optimization recommendations."""
        suggestions = self.optimizer.get_language_recommendations("nodejs")

        assert len(suggestions) > 0

        # Check for Node.js base image
        base_image_suggestions = [s for s in suggestions if s.type == "base_image"]
        assert len(base_image_suggestions) > 0
        assert "node:" in base_image_suggestions[0].description

        # Check for Node.js-specific optimizations
        optimization_texts = [s.description for s in suggestions]
        assert any("npm" in text for text in optimization_texts)
        assert any("node_modules" in text for text in optimization_texts)

    def test_go_recommendations(self):
        """Test Go-specific optimization recommendations."""
        suggestions = self.optimizer.get_language_recommendations("go")

        assert len(suggestions) > 0

        # Check for Go base image
        base_image_suggestions = [s for s in suggestions if s.type == "base_image"]
        assert len(base_image_suggestions) > 0
        assert "golang:" in base_image_suggestions[0].description

        # Check for Go-specific optimizations
        optimization_texts = [s.description for s in suggestions]
        assert any("go mod" in text for text in optimization_texts)
        assert any("static" in text or "CGO_ENABLED" in text for text in optimization_texts)

    def test_java_recommendations(self):
        """Test Java-specific optimization recommendations."""
        suggestions = self.optimizer.get_language_recommendations("java")

        assert len(suggestions) > 0

        # Check for Java base image
        base_image_suggestions = [s for s in suggestions if s.type == "base_image"]
        assert len(base_image_suggestions) > 0
        assert "openjdk:" in base_image_suggestions[0].description

        # Check for Java-specific optimizations
        optimization_texts = [s.description for s in suggestions]
        assert any("JRE" in text for text in optimization_texts)
        assert any("Maven" in text or "Gradle" in text for text in optimization_texts)

    def test_rust_recommendations(self):
        """Test Rust-specific optimization recommendations."""
        suggestions = self.optimizer.get_language_recommendations("rust")

        assert len(suggestions) > 0

        # Check for Rust base image
        base_image_suggestions = [s for s in suggestions if s.type == "base_image"]
        assert len(base_image_suggestions) > 0
        assert "rust:" in base_image_suggestions[0].description

        # Check for Rust-specific optimizations
        optimization_texts = [s.description for s in suggestions]
        assert any("cargo" in text for text in optimization_texts)
        assert any("--release" in text for text in optimization_texts)

    def test_unknown_language_recommendations(self):
        """Test recommendations for unknown languages."""
        suggestions = self.optimizer.get_language_recommendations("unknown")

        assert len(suggestions) > 0

        # Should provide generic suggestions
        assert suggestions[0].type == "generic"
        assert "multi-stage" in suggestions[0].description

    def test_framework_specific_recommendations(self):
        """Test framework-specific recommendations."""
        # Test Django recommendations
        django_suggestions = self.optimizer.get_language_recommendations("python", "django")
        django_texts = [s.description for s in django_suggestions]
        assert any("collectstatic" in text for text in django_texts)
        assert any("gunicorn" in text for text in django_texts)

        # Test Express recommendations
        express_suggestions = self.optimizer.get_language_recommendations("nodejs", "express")
        express_texts = [s.description for s in express_suggestions]
        assert any("PM2" in text for text in express_texts)

        # Test Spring recommendations
        spring_suggestions = self.optimizer.get_language_recommendations("java", "spring")
        spring_texts = [s.description for s in spring_suggestions]
        assert any("layered" in text for text in spring_texts)

    def test_optimization_profiles(self):
        """Test different optimization profiles."""
        prod_suggestions = self.optimizer.get_language_recommendations("python", profile="production")
        dev_suggestions = self.optimizer.get_language_recommendations("python", profile="development")

        # Both should have suggestions
        assert len(prod_suggestions) > 0
        assert len(dev_suggestions) > 0

        # Production should recommend slim images
        prod_base_image = [s for s in prod_suggestions if s.type == "base_image"][0]
        assert "slim" in prod_base_image.description or "alpine" in prod_base_image.description

    def test_dockerfile_optimization(self):
        """Test Dockerfile optimization for specific languages."""
        temp_dir = tempfile.mkdtemp()
        project_path = Path(temp_dir)

        # Create a basic Dockerfile
        original_dockerfile = """FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]"""

        try:
            optimized_dockerfile, applied_suggestions = self.optimizer.optimize_dockerfile_for_language(
                project_path, original_dockerfile, "python"
            )

            # Should have optimized the base image
            assert "python:" in optimized_dockerfile
            assert len(applied_suggestions) > 0

        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestAnalyzeProjectLanguage:
    """Test the analyze_project_language function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def _create_file(self, filename: str, content: str = ""):
        """Create a test file in the project directory."""
        file_path = self.project_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path

    def test_complete_analysis(self):
        """Test complete project analysis."""
        self._create_file("requirements.txt", "django>=4.0")
        self._create_file("manage.py", "#!/usr/bin/env python")

        result = analyze_project_language(self.project_path)

        assert result["language"] == "python"
        assert result["language_confidence"] > 0.8
        assert result["framework"] == "django"
        assert result["framework_confidence"] > 0.5
        assert result["recommendations_available"] is True
        assert isinstance(result["detected_files"], dict)

    def test_analysis_unknown_project(self):
        """Test analysis of unknown project types."""
        self._create_file("README.md", "# Unknown project")

        result = analyze_project_language(self.project_path)

        assert result["language"] == "unknown"
        assert result["language_confidence"] == 0.0
        assert result["framework"] is None
        assert result["recommendations_available"] is False

    def test_analysis_with_confidence_scores(self):
        """Test that analysis returns proper confidence scores."""
        # Create a strong Python indicator
        self._create_file("pyproject.toml", "[tool.poetry]\nname = 'test'")

        result = analyze_project_language(self.project_path)

        assert result["language"] == "python"
        assert 0.0 <= result["language_confidence"] <= 1.0

        if result["framework"]:
            assert 0.0 <= result["framework_confidence"] <= 1.0


class TestLanguageOptimizerIntegration:
    """Integration tests for language optimizer."""

    def test_end_to_end_optimization_workflow(self):
        """Test complete optimization workflow."""
        temp_dir = tempfile.mkdtemp()
        project_path = Path(temp_dir)

        try:
            # Create a Python/Django project
            (project_path / "requirements.txt").write_text("django>=4.0\npsycopg2>=2.8")
            (project_path / "manage.py").write_text("#!/usr/bin/env python")

            # Analyze the project
            analysis = analyze_project_language(project_path)

            # Get recommendations
            optimizer = LanguageOptimizer()
            suggestions = optimizer.get_language_recommendations(
                analysis["language"],
                analysis["framework"]
            )

            # Verify we got comprehensive recommendations
            assert len(suggestions) > 5  # Should have multiple suggestions

            suggestion_types = {s.type for s in suggestions}
            assert "base_image" in suggestion_types
            assert "build_optimization" in suggestion_types
            assert "optimization" in suggestion_types
            assert "security" in suggestion_types
            assert "framework_optimization" in suggestion_types

            # Verify Django-specific recommendations are present
            django_suggestions = [s for s in suggestions if "django" in s.description.lower()]
            assert len(django_suggestions) > 0

        finally:
            import shutil
            shutil.rmtree(temp_dir)

    def test_performance_with_large_project(self):
        """Test performance with a project containing many files."""
        temp_dir = tempfile.mkdtemp()
        project_path = Path(temp_dir)

        try:
            # Create many files to simulate a large project
            for i in range(100):
                (project_path / f"file_{i}.py").write_text(f"# File {i}")

            # Add key indicator files
            (project_path / "requirements.txt").write_text("django>=4.0")
            (project_path / "manage.py").write_text("#!/usr/bin/env python")

            # Analysis should still be fast and accurate
            import time
            start_time = time.time()

            analysis = analyze_project_language(project_path)

            end_time = time.time()

            # Should complete quickly (under 1 second for 100 files)
            assert (end_time - start_time) < 1.0

            # Should still detect correctly
            assert analysis["language"] == "python"
            assert analysis["framework"] == "django"

        finally:
            import shutil
            shutil.rmtree(temp_dir)
