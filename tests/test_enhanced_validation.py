"""Tests for enhanced validation system."""

import pytest
from pathlib import Path

from src.docker_optimizer.enhanced_validation import (
    EnhancedValidator,
    ValidationLevel,
    ValidationScope,
    ValidationRule,
    DockerfileInstruction,
    SecurityValidationConfig,
    PerformanceValidationConfig,
    ErrorSeverity
)


class TestDockerfileInstruction:
    """Test DockerfileInstruction model."""
    
    def test_valid_instruction(self):
        """Test valid Dockerfile instruction."""
        instruction = DockerfileInstruction(
            command="FROM",
            arguments="ubuntu:20.04",
            line_number=1,
            original_line="FROM ubuntu:20.04"
        )
        
        assert instruction.command == "FROM"
        assert instruction.arguments == "ubuntu:20.04"
        assert instruction.line_number == 1
    
    def test_invalid_command(self):
        """Test invalid Dockerfile command."""
        with pytest.raises(ValueError, match="Invalid Dockerfile command"):
            DockerfileInstruction(
                command="INVALID_COMMAND",
                arguments="some args",
                line_number=1,
                original_line="INVALID_COMMAND some args"
            )
    
    def test_command_case_normalization(self):
        """Test command case normalization."""
        instruction = DockerfileInstruction(
            command="from",
            arguments="ubuntu:20.04",
            line_number=1,
            original_line="from ubuntu:20.04"
        )
        
        assert instruction.command == "FROM"


class TestValidationRule:
    """Test ValidationRule functionality."""
    
    def test_validation_rule_creation(self):
        """Test validation rule creation."""
        rule = ValidationRule(
            id="TEST_001",
            name="Test Rule",
            description="A test validation rule",
            scope=ValidationScope.SYNTAX,
            severity=ErrorSeverity.HIGH,
            pattern=r"FROM\\s+.*:latest",
            fix_suggestion="Use specific version tags"
        )
        
        assert rule.id == "TEST_001"
        assert rule.scope == ValidationScope.SYNTAX
        assert rule.severity == ErrorSeverity.HIGH
        assert rule.enabled


class TestSecurityValidationConfig:
    """Test SecurityValidationConfig."""
    
    def test_default_config(self):
        """Test default security validation configuration."""
        config = SecurityValidationConfig()
        
        assert config.check_root_user
        assert config.check_package_versions
        assert config.check_secrets
        assert config.check_privileged_commands
        assert len(config.secret_patterns) > 0
    
    def test_custom_config(self):
        """Test custom security validation configuration."""
        config = SecurityValidationConfig(
            check_root_user=False,
            allowed_base_images=["ubuntu", "alpine"],
            blocked_packages=["telnet", "ftp"]
        )
        
        assert not config.check_root_user
        assert "ubuntu" in config.allowed_base_images
        assert "telnet" in config.blocked_packages


class TestPerformanceValidationConfig:
    """Test PerformanceValidationConfig."""
    
    def test_default_config(self):
        """Test default performance validation configuration."""
        config = PerformanceValidationConfig()
        
        assert config.max_layers == 20
        assert config.check_cache_optimization
        assert config.check_multi_stage
        assert config.max_image_size_mb == 1000
    
    def test_custom_config(self):
        """Test custom performance validation configuration."""
        config = PerformanceValidationConfig(
            max_layers=15,
            max_image_size_mb=500,
            check_unnecessary_packages=False
        )
        
        assert config.max_layers == 15
        assert config.max_image_size_mb == 500
        assert not config.check_unnecessary_packages


class TestEnhancedValidator:
    """Test EnhancedValidator functionality."""
    
    def setup_method(self):
        """Setup test method."""
        self.validator = EnhancedValidator(level=ValidationLevel.STANDARD)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = EnhancedValidator()
        
        assert validator.level == ValidationLevel.STANDARD
        assert len(validator.validation_rules) > 0
        assert validator.security_config is not None
        assert validator.performance_config is not None
    
    def test_parse_instructions_basic(self):
        """Test basic instruction parsing."""
        content = '''FROM ubuntu:20.04
RUN apt-get update
COPY . /app
WORKDIR /app'''
        
        lines = content.split('\\n')
        instructions = self.validator._parse_instructions(lines)
        
        assert len(instructions) == 4
        assert instructions[0].command == "FROM"
        assert instructions[0].arguments == "ubuntu:20.04"
        assert instructions[1].command == "RUN"
        assert instructions[2].command == "COPY"
        assert instructions[3].command == "WORKDIR"
    
    def test_parse_instructions_multiline(self):
        """Test multiline instruction parsing."""
        content = '''FROM ubuntu:20.04
RUN apt-get update && \\\\
    apt-get install -y python3 && \\\\
    rm -rf /var/lib/apt/lists/*'''
        
        lines = content.split('\\n')
        instructions = self.validator._parse_instructions(lines)
        
        assert len(instructions) == 2
        assert instructions[1].command == "RUN"
        assert "apt-get update" in instructions[1].arguments
        assert "rm -rf" in instructions[1].arguments
    
    def test_parse_instructions_with_comments(self):
        """Test parsing with comments and empty lines."""
        content = '''# Base image
FROM ubuntu:20.04

# Install dependencies
RUN apt-get update
'''
        
        lines = content.split('\\n')
        instructions = self.validator._parse_instructions(lines)
        
        assert len(instructions) == 2
        assert instructions[0].command == "FROM"
        assert instructions[1].command == "RUN"
    
    def test_validate_syntax_success(self):
        """Test successful syntax validation."""
        instructions = [
            DockerfileInstruction("FROM", "ubuntu:20.04", 1, "FROM ubuntu:20.04"),
            DockerfileInstruction("RUN", "apt-get update", 2, "RUN apt-get update")
        ]
        
        result = self.validator._validate_syntax(instructions)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_syntax_no_from(self):
        """Test syntax validation with missing FROM."""
        instructions = [
            DockerfileInstruction("RUN", "apt-get update", 1, "RUN apt-get update")
        ]
        
        result = self.validator._validate_syntax(instructions)
        
        assert not result.is_valid
        assert "First instruction must be FROM" in result.errors
    
    def test_validate_syntax_empty(self):
        """Test syntax validation with empty Dockerfile."""
        instructions = []
        
        result = self.validator._validate_syntax(instructions)
        
        assert not result.is_valid
        assert "empty" in result.errors[0].lower()
    
    def test_validate_semantics(self):
        """Test semantic validation."""
        instructions = [
            DockerfileInstruction("FROM", "ubuntu:20.04", 1, "FROM ubuntu:20.04"),
            DockerfileInstruction("COPY", ". /app", 2, "COPY . /app"),
            DockerfileInstruction("WORKDIR", "/app", 3, "WORKDIR /app")
        ]
        
        result = self.validator._validate_semantics(instructions)
        
        assert result.is_valid  # Should be valid
        # Should suggest WORKDIR before COPY
        assert any("WORKDIR" in suggestion for suggestion in result.suggestions)
    
    def test_validate_security_root_user(self):
        """Test security validation for root user."""
        content = '''FROM ubuntu:20.04
RUN apt-get update
CMD ["python", "app.py"]'''
        
        instructions = self.validator._parse_instructions(content.split('\\n'))
        result = self.validator._validate_security(content, instructions)
        
        # Should warn about no USER instruction
        assert any("root" in warning.lower() for warning in result.warnings)
        assert any("USER" in suggestion for suggestion in result.suggestions)
    
    def test_validate_security_secrets(self):
        """Test security validation for secrets."""
        content = '''FROM ubuntu:20.04
ENV PASSWORD=secret123
RUN echo "api_key=abc123" > config'''
        
        instructions = self.validator._parse_instructions(content.split('\\n'))
        result = self.validator._validate_security(content, instructions)
        
        # Should detect potential secrets
        assert len(result.errors) > 0
        assert any("secret" in error.lower() for error in result.errors)
    
    def test_validate_security_privileged_commands(self):
        """Test security validation for privileged commands."""
        content = '''FROM ubuntu:20.04
RUN sudo apt-get update
RUN chmod 777 /app'''
        
        instructions = self.validator._parse_instructions(content.split('\\n'))
        result = self.validator._validate_security(content, instructions)
        
        # Should warn about dangerous commands
        assert len(result.warnings) > 0
        assert any("dangerous" in warning.lower() for warning in result.warnings)
    
    def test_validate_performance_layers(self):
        """Test performance validation for layer count."""
        # Create many instructions to exceed layer limit
        instructions = []
        for i in range(25):  # More than default max_layers (20)
            instructions.append(
                DockerfileInstruction("RUN", f"echo {i}", i+1, f"RUN echo {i}")
            )
        
        result = self.validator._validate_performance(instructions)
        
        # Should warn about too many layers
        assert any("layer" in warning.lower() for warning in result.warnings)
    
    def test_validate_performance_cache_cleanup(self):
        """Test performance validation for cache cleanup."""
        instructions = [
            DockerfileInstruction("FROM", "ubuntu:20.04", 1, "FROM ubuntu:20.04"),
            DockerfileInstruction("RUN", "apt-get update", 2, "RUN apt-get update"),
            DockerfileInstruction("RUN", "apt-get install -y python3", 3, "RUN apt-get install -y python3")
        ]
        
        result = self.validator._validate_performance(instructions)
        
        # Should suggest cache cleanup
        assert any("cache" in suggestion.lower() for suggestion in result.suggestions)
    
    def test_validate_best_practices_latest_tag(self):
        """Test best practices validation for latest tag."""
        content = '''FROM ubuntu:latest
RUN apt-get update'''
        
        instructions = self.validator._parse_instructions(content.split('\\n'))
        result = self.validator._validate_best_practices(content, instructions)
        
        # Should warn about latest tag
        assert any("latest" in warning.lower() for warning in result.warnings)
        assert any("specific" in suggestion.lower() for suggestion in result.suggestions)
    
    def test_validate_best_practices_labels(self):
        """Test best practices validation for labels."""
        content = '''FROM ubuntu:20.04
RUN apt-get update
CMD ["python", "app.py"]'''
        
        instructions = self.validator._parse_instructions(content.split('\\n'))
        result = self.validator._validate_best_practices(content, instructions)
        
        # Should suggest adding labels
        assert any("label" in suggestion.lower() for suggestion in result.suggestions)
    
    def test_validate_best_practices_multistage(self):
        """Test best practices validation for multi-stage builds."""
        content = '''FROM ubuntu:20.04
RUN apt-get update && apt-get install -y gcc make
COPY . /app
WORKDIR /app
RUN make build
CMD ["./app"]'''
        
        instructions = self.validator._parse_instructions(content.split('\\n'))
        result = self.validator._validate_best_practices(content, instructions)
        
        # Should suggest multi-stage build
        assert any("multi-stage" in suggestion.lower() for suggestion in result.suggestions)
    
    def test_validate_dockerfile_content_complete(self):
        """Test complete Dockerfile validation."""
        content = '''FROM ubuntu:latest
RUN apt-get update
COPY . /app
CMD ["python", "app.py"]'''
        
        result = self.validator.validate_dockerfile_content(content)
        
        # Should have various warnings and suggestions
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.suggestions, list)
        
        # Should be recorded in history
        assert len(self.validator.validation_history) > 0
    
    def test_validate_configuration_valid(self):
        """Test configuration validation with valid config."""
        config = {
            "optimization_level": "standard",
            "security_scan": True,
            "max_layers": 15,
            "timeout_seconds": 300
        }
        
        result = self.validator.validate_configuration(config)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_configuration_missing_required(self):
        """Test configuration validation with missing required fields."""
        config = {
            "security_scan": True
            # Missing optimization_level
        }
        
        result = self.validator.validate_configuration(config)
        
        assert not result.is_valid
        assert any("optimization_level" in error for error in result.errors)
    
    def test_validate_configuration_invalid_values(self):
        """Test configuration validation with invalid values."""
        config = {
            "optimization_level": "invalid_level",
            "security_scan": True,
            "max_layers": 100,  # Too high
            "timeout_seconds": 5  # Too low
        }
        
        result = self.validator.validate_configuration(config)
        
        assert not result.is_valid
        assert len(result.errors) >= 3  # Invalid level, max_layers, timeout
    
    def test_get_validation_report(self):
        """Test validation report generation."""
        # Perform some validations first
        self.validator.validate_dockerfile_content("FROM ubuntu:20.04\\nRUN echo test")
        
        report = self.validator.get_validation_report()
        
        assert "summary" in report
        assert "total_validations" in report["summary"]
        assert "severity_distribution" in report
        assert "validation_level" in report
        assert report["validation_level"] == ValidationLevel.STANDARD.value
    
    def test_get_validation_report_no_history(self):
        """Test validation report with no history."""
        validator = EnhancedValidator()
        report = validator.get_validation_report()
        
        assert "message" in report
        assert "No validations performed" in report["message"]


class TestValidationLevels:
    """Test different validation levels."""
    
    def test_basic_level(self):
        """Test basic validation level."""
        validator = EnhancedValidator(level=ValidationLevel.BASIC)
        assert validator.level == ValidationLevel.BASIC
    
    def test_strict_level(self):
        """Test strict validation level."""
        validator = EnhancedValidator(level=ValidationLevel.STRICT)
        assert validator.level == ValidationLevel.STRICT
    
    def test_enterprise_level(self):
        """Test enterprise validation level."""
        validator = EnhancedValidator(level=ValidationLevel.ENTERPRISE)
        assert validator.level == ValidationLevel.ENTERPRISE


class TestCustomConfigurations:
    """Test custom validation configurations."""
    
    def test_custom_security_config(self):
        """Test custom security configuration."""
        security_config = SecurityValidationConfig(
            check_root_user=False,
            allowed_base_images=["alpine", "ubuntu"],
            secret_patterns=["custom_secret_pattern"]
        )
        
        validator = EnhancedValidator(security_config=security_config)
        
        assert validator.security_config.allowed_base_images == ["alpine", "ubuntu"]
        assert not validator.security_config.check_root_user
        assert validator.security_config.secret_patterns == ["custom_secret_pattern"]
    
    def test_custom_performance_config(self):
        """Test custom performance configuration."""
        performance_config = PerformanceValidationConfig(
            max_layers=10,
            max_image_size_mb=200,
            check_cache_optimization=False
        )
        
        validator = EnhancedValidator(performance_config=performance_config)
        
        assert validator.performance_config.max_layers == 10
        assert validator.performance_config.max_image_size_mb == 200
        assert not validator.performance_config.check_cache_optimization


@pytest.mark.integration
class TestValidatorIntegration:
    """Integration tests for the validation system."""
    
    def test_comprehensive_dockerfile_validation(self):
        """Test comprehensive Dockerfile validation."""
        # Complex Dockerfile with various issues
        complex_dockerfile = '''# Multi-stage Dockerfile with issues
FROM node:latest as builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:latest
RUN apt-get update && apt-get install -y curl
ENV API_KEY=secret123
COPY --from=builder /app/dist /usr/share/nginx/html
RUN chmod 777 /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
'''
        
        validator = EnhancedValidator(level=ValidationLevel.STRICT)
        result = validator.validate_dockerfile_content(complex_dockerfile)
        
        # Should detect multiple issues
        assert len(result.warnings) > 0  # Latest tags, no USER instruction
        assert len(result.errors) > 0   # Secrets detected
        assert len(result.suggestions) > 0  # Various improvements
        
        # Verify specific issues are caught
        all_issues = result.errors + result.warnings + result.suggestions
        issues_text = " ".join(all_issues).lower()
        
        assert "latest" in issues_text  # Latest tag usage
        assert "secret" in issues_text or "credential" in issues_text  # Secret detection
        assert "777" in issues_text or "dangerous" in issues_text  # Dangerous chmod
    
    def test_validation_with_good_dockerfile(self):
        """Test validation with a well-written Dockerfile."""
        good_dockerfile = '''# Multi-stage build for Node.js app
FROM node:16-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:16-alpine AS runtime
RUN addgroup -g 1001 -S nodejs && \\
    adduser -S nextjs -u 1001
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .
USER nextjs
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:3000/health || exit 1
CMD ["node", "server.js"]
'''
        
        validator = EnhancedValidator(level=ValidationLevel.STANDARD)
        result = validator.validate_dockerfile_content(good_dockerfile)
        
        # Should have minimal issues
        assert len(result.errors) == 0  # No critical errors
        # May have some suggestions for further improvements
        
        # Should be marked as valid overall
        assert result.is_valid or len(result.errors) == 0