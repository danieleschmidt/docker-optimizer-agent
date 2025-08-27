"""Enhanced validation system for Docker Optimizer Agent.

Comprehensive validation framework with schema validation, semantic analysis,
security checks, and best practices enforcement.
"""

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, ValidationError, validator

from .enhanced_error_handling import ErrorSeverity


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    ENTERPRISE = "enterprise"


class ValidationScope(Enum):
    """Validation scope areas."""
    SYNTAX = "syntax"
    SEMANTICS = "semantics"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BEST_PRACTICES = "best_practices"
    COMPLIANCE = "compliance"


@dataclass
class ValidationRule:
    """Individual validation rule definition."""
    id: str
    name: str
    description: str
    scope: ValidationScope
    severity: ErrorSeverity
    pattern: Optional[str] = None
    validator_func: Optional[callable] = None
    fix_suggestion: Optional[str] = None
    enabled: bool = True


class DockerfileInstruction(BaseModel):
    """Validated Dockerfile instruction model."""
    command: str
    arguments: str
    line_number: int
    original_line: str
    
    @validator('command')
    def command_must_be_valid(cls, v):
        valid_commands = {
            'FROM', 'RUN', 'CMD', 'LABEL', 'MAINTAINER', 'EXPOSE', 
            'ENV', 'ADD', 'COPY', 'ENTRYPOINT', 'VOLUME', 'USER',
            'WORKDIR', 'ARG', 'ONBUILD', 'STOPSIGNAL', 'HEALTHCHECK',
            'SHELL'
        }
        if v.upper() not in valid_commands:
            raise ValueError(f'Invalid Dockerfile command: {v}')
        return v.upper()


class SecurityValidationConfig(BaseModel):
    """Security validation configuration."""
    check_root_user: bool = True
    check_package_versions: bool = True
    check_secrets: bool = True
    check_privileged_commands: bool = True
    allowed_base_images: List[str] = []
    blocked_packages: List[str] = []
    secret_patterns: List[str] = field(default_factory=lambda: [
        r'password\s*=\s*["\']?\w+["\']?',
        r'api_key\s*=\s*["\']?\w+["\']?',
        r'secret\s*=\s*["\']?\w+["\']?',
        r'token\s*=\s*["\']?\w+["\']?',
        r'-----BEGIN\s+(PRIVATE\s+KEY|RSA\s+PRIVATE\s+KEY)',
    ])


class PerformanceValidationConfig(BaseModel):
    """Performance validation configuration."""
    max_layers: int = 20
    check_cache_optimization: bool = True
    check_multi_stage: bool = True
    max_image_size_mb: int = 1000
    check_unnecessary_packages: bool = True


class EnhancedValidator:
    """Comprehensive validation system for Docker optimization."""
    
    def __init__(
        self,
        level: ValidationLevel = ValidationLevel.STANDARD,
        security_config: Optional[SecurityValidationConfig] = None,
        performance_config: Optional[PerformanceValidationConfig] = None
    ):
        self.level = level
        self.security_config = security_config or SecurityValidationConfig()
        self.performance_config = performance_config or PerformanceValidationConfig()
        self.validation_rules = self._initialize_rules()
        self.validation_history: List[ValidationResult] = []
    
    def _initialize_rules(self) -> List[ValidationRule]:
        """Initialize validation rules based on configuration."""
        rules = []
        
        # Syntax Rules
        rules.extend([
            ValidationRule(
                id="SYNTAX_001",
                name="Valid FROM instruction",
                description="FROM instruction must be present and valid",
                scope=ValidationScope.SYNTAX,
                severity=ErrorSeverity.CRITICAL,
                pattern=r'^FROM\s+\S+',
                fix_suggestion="Add a valid FROM instruction with a base image"
            ),
            ValidationRule(
                id="SYNTAX_002", 
                name="No empty lines in commands",
                description="Multi-line commands should not have empty lines",
                scope=ValidationScope.SYNTAX,
                severity=ErrorSeverity.LOW,
                pattern=r'\\\\$\\n\\s*$',
                fix_suggestion="Remove empty lines in multi-line commands"
            ),
        ])
        
        # Security Rules
        if self.security_config.check_root_user:
            rules.append(ValidationRule(
                id="SECURITY_001",
                name="Non-root user required",
                description="Container should not run as root user",
                scope=ValidationScope.SECURITY,
                severity=ErrorSeverity.HIGH,
                fix_suggestion="Add 'USER non-root-user' instruction"
            ))
        
        if self.security_config.check_secrets:
            for i, pattern in enumerate(self.security_config.secret_patterns):
                rules.append(ValidationRule(
                    id=f"SECURITY_SECRET_{i:03d}",
                    name=f"Secret pattern detection {i+1}",
                    description="Potential secret or credential found in Dockerfile",
                    scope=ValidationScope.SECURITY,
                    severity=ErrorSeverity.CRITICAL,
                    pattern=pattern,
                    fix_suggestion="Use build-time secrets or environment variables"
                ))
        
        # Performance Rules
        if self.performance_config.check_cache_optimization:
            rules.append(ValidationRule(
                id="PERFORMANCE_001",
                name="Package cache cleanup",
                description="Package manager caches should be cleaned",
                scope=ValidationScope.PERFORMANCE,
                severity=ErrorSeverity.MEDIUM,
                fix_suggestion="Add cache cleanup commands (e.g., 'rm -rf /var/lib/apt/lists/*')"
            ))
        
        # Best Practices Rules
        rules.extend([
            ValidationRule(
                id="BEST_PRACTICE_001",
                name="Specific image tags",
                description="Avoid using 'latest' tag for base images",
                scope=ValidationScope.BEST_PRACTICES,
                severity=ErrorSeverity.MEDIUM,
                pattern=r':latest\\b',
                fix_suggestion="Use specific version tags instead of 'latest'"
            ),
            ValidationRule(
                id="BEST_PRACTICE_002",
                name="Label metadata",
                description="Include descriptive labels for image metadata",
                scope=ValidationScope.BEST_PRACTICES,
                severity=ErrorSeverity.LOW,
                fix_suggestion="Add LABEL instructions for maintainer, version, description"
            ),
        ])
        
        return rules
    
    def validate_dockerfile_content(self, content: str) -> ValidationResult:
        """Validate Dockerfile content comprehensively."""
        errors = []
        warnings = []
        suggestions = []
        
        lines = content.strip().split('\\n')
        instructions = self._parse_instructions(lines)
        
        # Validate instruction syntax
        syntax_result = self._validate_syntax(instructions)
        errors.extend(syntax_result.errors)
        warnings.extend(syntax_result.warnings)
        suggestions.extend(syntax_result.suggestions)
        
        # Validate semantics
        semantic_result = self._validate_semantics(instructions)
        errors.extend(semantic_result.errors)
        warnings.extend(semantic_result.warnings)
        suggestions.extend(semantic_result.suggestions)
        
        # Security validation
        security_result = self._validate_security(content, instructions)
        errors.extend(security_result.errors)
        warnings.extend(security_result.warnings)
        suggestions.extend(security_result.suggestions)
        
        # Performance validation
        performance_result = self._validate_performance(instructions)
        errors.extend(performance_result.errors)
        warnings.extend(performance_result.warnings)
        suggestions.extend(performance_result.suggestions)
        
        # Best practices validation
        practices_result = self._validate_best_practices(content, instructions)
        errors.extend(practices_result.errors)
        warnings.extend(practices_result.warnings)
        suggestions.extend(practices_result.suggestions)
        
        # Determine overall severity
        severity = ErrorSeverity.LOW
        if errors:
            severity = ErrorSeverity.HIGH
        elif warnings:
            severity = ErrorSeverity.MEDIUM
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            severity=severity
        )
        
        self.validation_history.append(result)
        return result
    
    def _parse_instructions(self, lines: List[str]) -> List[DockerfileInstruction]:
        """Parse Dockerfile lines into structured instructions."""
        instructions = []
        current_instruction = None
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check if this is a continuation of a multi-line instruction
            if line.endswith('\\\\'):
                if current_instruction:
                    current_instruction.arguments += ' ' + line[:-1].strip()
                else:
                    # New multi-line instruction
                    parts = line[:-1].strip().split(None, 1)
                    if len(parts) >= 2:
                        command, args = parts[0], parts[1]
                    else:
                        command, args = parts[0], ''
                    
                    try:
                        current_instruction = DockerfileInstruction(
                            command=command,
                            arguments=args,
                            line_number=i,
                            original_line=line
                        )
                    except ValidationError as e:
                        # Handle invalid instruction
                        continue
            else:
                # Complete instruction (single line or end of multi-line)
                if current_instruction:
                    current_instruction.arguments += ' ' + line
                    instructions.append(current_instruction)
                    current_instruction = None
                else:
                    # Single line instruction
                    parts = line.split(None, 1)
                    if len(parts) >= 2:
                        command, args = parts[0], parts[1]
                    else:
                        command, args = parts[0], ''
                    
                    try:
                        instruction = DockerfileInstruction(
                            command=command,
                            arguments=args,
                            line_number=i,
                            original_line=line
                        )
                        instructions.append(instruction)
                    except ValidationError:
                        # Handle invalid instruction
                        continue
        
        return instructions
    
    def _validate_syntax(self, instructions: List[DockerfileInstruction]) -> ValidationResult:
        """Validate Dockerfile syntax."""
        errors = []
        warnings = []
        suggestions = []
        
        if not instructions:
            errors.append("Dockerfile is empty or contains no valid instructions")
            return ValidationResult(False, errors, warnings, suggestions)
        
        # Check if FROM is first instruction
        first_instruction = instructions[0]
        if first_instruction.command != 'FROM':
            errors.append("First instruction must be FROM")
        
        # Check for duplicate instructions that should be unique
        unique_commands = {'FROM', 'MAINTAINER'}
        seen_commands = set()
        
        for instruction in instructions:
            if instruction.command in unique_commands:
                if instruction.command in seen_commands:
                    warnings.append(f"Multiple {instruction.command} instructions found")
                seen_commands.add(instruction.command)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_semantics(self, instructions: List[DockerfileInstruction]) -> ValidationResult:
        """Validate Dockerfile semantics and instruction relationships."""
        errors = []
        warnings = []
        suggestions = []
        
        # Check for WORKDIR before COPY/ADD operations
        workdir_set = False
        for instruction in instructions:
            if instruction.command == 'WORKDIR':
                workdir_set = True
            elif instruction.command in ['COPY', 'ADD'] and not workdir_set:
                suggestions.append("Consider setting WORKDIR before COPY/ADD operations")
                break
        
        # Check for health checks in web applications
        has_expose = any(inst.command == 'EXPOSE' for inst in instructions)
        has_healthcheck = any(inst.command == 'HEALTHCHECK' for inst in instructions)
        
        if has_expose and not has_healthcheck:
            suggestions.append("Consider adding HEALTHCHECK for exposed services")
        
        # Check for proper USER instruction placement
        user_instructions = [inst for inst in instructions if inst.command == 'USER']
        if len(user_instructions) > 1:
            warnings.append("Multiple USER instructions may indicate security issues")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_security(self, content: str, instructions: List[DockerfileInstruction]) -> ValidationResult:
        """Validate security aspects of Dockerfile."""
        errors = []
        warnings = []
        suggestions = []
        
        # Check for root user
        if self.security_config.check_root_user:
            has_user_instruction = any(inst.command == 'USER' for inst in instructions)
            if not has_user_instruction:
                warnings.append("No USER instruction found - container will run as root")
                suggestions.append("Add 'USER non-root-user' instruction for security")
        
        # Check for secrets in content
        if self.security_config.check_secrets:
            for pattern in self.security_config.secret_patterns:
                if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                    errors.append("Potential secret or credential found in Dockerfile")
                    suggestions.append("Use build secrets or environment variables instead")
                    break
        
        # Check for privileged operations
        if self.security_config.check_privileged_commands:
            dangerous_patterns = [
                r'sudo\\s+',
                r'chmod\\s+777',
                r'--privileged',
                r'--user\\s+root',
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    warnings.append(f"Potentially dangerous command pattern detected")
                    suggestions.append("Review security implications of privileged commands")
        
        # Check base image against allowed list
        from_instructions = [inst for inst in instructions if inst.command == 'FROM']
        if from_instructions and self.security_config.allowed_base_images:
            base_image = from_instructions[0].arguments.split()[0]
            if not any(allowed in base_image for allowed in self.security_config.allowed_base_images):
                warnings.append(f"Base image '{base_image}' is not in allowed list")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_performance(self, instructions: List[DockerfileInstruction]) -> ValidationResult:
        """Validate performance aspects of Dockerfile."""
        errors = []
        warnings = []
        suggestions = []
        
        # Check layer count
        if len(instructions) > self.performance_config.max_layers:
            warnings.append(f"High number of layers ({len(instructions)}) may impact performance")
            suggestions.append("Consider combining RUN instructions to reduce layers")
        
        # Check for cache cleanup
        run_instructions = [inst for inst in instructions if inst.command == 'RUN']
        apt_update_found = False
        apt_clean_found = False
        
        for instruction in run_instructions:
            args = instruction.arguments
            if 'apt-get update' in args or 'apt update' in args:
                apt_update_found = True
            if any(pattern in args for pattern in ['apt-get clean', 'rm -rf /var/lib/apt/lists/*']):
                apt_clean_found = True
        
        if apt_update_found and not apt_clean_found:
            suggestions.append("Clean package manager caches to reduce image size")
        
        # Check for unnecessary packages
        if self.performance_config.check_unnecessary_packages:
            unnecessary_patterns = [
                r'vim\\b', r'nano\\b', r'curl\\b.*wget\\b', r'git\\b.*subversion\\b'
            ]
            
            for instruction in run_instructions:
                for pattern in unnecessary_patterns:
                    if re.search(pattern, instruction.arguments, re.IGNORECASE):
                        suggestions.append("Consider removing unnecessary packages to reduce image size")
                        break
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_best_practices(self, content: str, instructions: List[DockerfileInstruction]) -> ValidationResult:
        """Validate adherence to Dockerfile best practices."""
        errors = []
        warnings = []
        suggestions = []
        
        # Check for specific tags
        from_instructions = [inst for inst in instructions if inst.command == 'FROM']
        for instruction in from_instructions:
            if ':latest' in instruction.arguments or ':' not in instruction.arguments:
                warnings.append("Using 'latest' tag or no tag specification")
                suggestions.append("Use specific version tags for reproducible builds")
        
        # Check for metadata labels
        has_labels = any(inst.command == 'LABEL' for inst in instructions)
        if not has_labels:
            suggestions.append("Add LABEL instructions for better image metadata")
        
        # Check for multi-stage builds opportunity
        from_count = len(from_instructions)
        if from_count == 1:
            # Look for build tools that suggest multi-stage opportunity
            build_tools = ['gcc', 'make', 'cmake', 'maven', 'gradle', 'npm install', 'pip install']
            has_build_tools = any(
                any(tool in inst.arguments.lower() for tool in build_tools)
                for inst in instructions if inst.command == 'RUN'
            )
            
            if has_build_tools:
                suggestions.append("Consider using multi-stage build to reduce final image size")
        
        # Check for proper signal handling
        has_stopsignal = any(inst.command == 'STOPSIGNAL' for inst in instructions)
        has_entrypoint = any(inst.command == 'ENTRYPOINT' for inst in instructions)
        
        if has_entrypoint and not has_stopsignal:
            suggestions.append("Consider adding STOPSIGNAL for proper container shutdown")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def validate_configuration(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate optimizer configuration."""
        errors = []
        warnings = []
        suggestions = []
        
        required_fields = ['optimization_level', 'security_scan']
        for field in required_fields:
            if field not in config:
                errors.append(f"Required configuration field missing: {field}")
        
        # Validate optimization level
        if 'optimization_level' in config:
            valid_levels = ['basic', 'standard', 'aggressive']
            if config['optimization_level'] not in valid_levels:
                errors.append(f"Invalid optimization level: {config['optimization_level']}")
        
        # Validate numeric ranges
        numeric_validations = {
            'max_layers': (1, 50),
            'timeout_seconds': (10, 3600),
            'max_image_size_mb': (10, 10000)
        }
        
        for field, (min_val, max_val) in numeric_validations.items():
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    errors.append(f"Invalid {field}: must be between {min_val} and {max_val}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        if not self.validation_history:
            return {"message": "No validations performed yet"}
        
        recent_validations = self.validation_history[-10:]  # Last 10 validations
        
        total_errors = sum(len(v.errors) for v in recent_validations)
        total_warnings = sum(len(v.warnings) for v in recent_validations)
        total_suggestions = sum(len(v.suggestions) for v in recent_validations)
        
        severity_counts = {
            severity.value: sum(1 for v in recent_validations if v.severity == severity)
            for severity in ErrorSeverity
        }
        
        return {
            "summary": {
                "total_validations": len(recent_validations),
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "total_suggestions": total_suggestions,
                "success_rate": sum(1 for v in recent_validations if v.is_valid) / len(recent_validations)
            },
            "severity_distribution": severity_counts,
            "validation_level": self.level.value,
            "enabled_scopes": [scope.value for scope in ValidationScope],
            "timestamp": time.time()
        }