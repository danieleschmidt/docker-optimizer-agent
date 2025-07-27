"""Configuration management for Docker Optimizer Agent."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration-related errors."""

    def __init__(self, message: str, field_path: Optional[str] = None, suggestions: Optional[List[str]] = None):
        """Initialize ConfigError with detailed context.
        
        Args:
            message: Error message
            field_path: Path to the problematic field (e.g., 'cache_settings.max_size')
            suggestions: List of suggested fixes
        """
        super().__init__(message)
        self.field_path = field_path
        self.suggestions = suggestions or []

    def __str__(self) -> str:
        msg = super().__str__()
        if self.field_path:
            msg = f"Configuration error in '{self.field_path}': {msg}"
        if self.suggestions:
            msg += "\nSuggestions:\n" + "\n".join(f"  - {s}" for s in self.suggestions)
        return msg


class Config:
    """Configuration management for Docker Optimizer Agent.

    Supports loading configuration from:
    1. Default values (built-in)
    2. Configuration files (YAML/JSON)
    3. Environment variables (highest priority)

    Configuration file locations (in order):
    1. ~/.docker-optimizer.yml
    2. ~/.docker-optimizer.yaml
    3. ~/.docker-optimizer.json
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_path: Optional path to configuration file
        """
        self._config = self._load_default_config()

        # Try to load user config from home directory
        if config_path is None:
            config_path = self._find_user_config()

        if config_path and Path(config_path).exists():
            try:
                user_config = self._load_config_file(config_path)
                self._merge_config(user_config)
                logger.info("Loaded configuration from %s", config_path)
            except ConfigError:
                raise  # Re-raise ConfigError with full context
            except Exception as e:
                raise ConfigError(
                    f"Failed to load configuration from {config_path}: {e}",
                    suggestions=[
                        "Check that the file exists and is readable",
                        "Verify the file format is valid YAML or JSON",
                        "Check file permissions"
                    ]
                )

        # Apply environment variable overrides
        self._apply_env_overrides()

        # Validate final configuration
        validation_errors = self.validate_config()
        if validation_errors:
            raise ConfigError(
                "Configuration validation failed after loading",
                suggestions=["Fix the following issues:"] + validation_errors
            )

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Create configuration from specific file.

        Args:
            config_path: Path to configuration file

        Returns:
            Config: Configuration instance

        Raises:
            ConfigError: If file cannot be loaded
        """
        if not Path(config_path).exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        return cls(config_path=config_path)

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            "base_image_sizes": {
                # Base Linux distributions (MB)
                "alpine:latest": 5,
                "alpine:3.18": 5,
                "alpine:3.17": 5,
                "alpine": 5,
                "ubuntu:22.04": 77,
                "ubuntu:20.04": 72,
                "ubuntu:18.04": 63,
                "ubuntu:latest": 77,
                "ubuntu": 77,
                "debian:12": 117,
                "debian:11": 124,
                "debian:12-slim": 74,
                "debian:11-slim": 80,
                "debian:latest": 117,
                "debian": 117,
                # Language-specific images (MB)
                "python:3.11": 1013,
                "python:3.11-slim": 130,
                "python:3.11-alpine": 47,
                "python:3.10": 995,
                "python:3.10-slim": 125,
                "python:3.10-alpine": 45,
                "python": 1013,
                "node:18": 993,
                "node:18-slim": 167,
                "node:18-alpine": 110,
                "node:16": 943,
                "node:16-slim": 159,
                "node:16-alpine": 109,
                "node": 993,
                "golang:1.20": 964,
                "golang:1.20-alpine": 107,
                "golang": 964,
                "openjdk:17": 471,
                "openjdk:11": 628,
                "openjdk": 471,
                "rust:1.70": 1500,
                "rust:1.70-slim": 700,
                "rust": 1500,
            },
            "package_sizes": {
                # Common package sizes (MB)
                "curl": 8,
                "wget": 3,
                "git": 12,
                "vim": 15,
                "nano": 2,
                "htop": 1,
                "ca-certificates": 5,
                "build-essential": 200,
                "gcc": 180,
                "make": 5,
                "python3": 25,
                "python3-pip": 50,
                "nodejs": 30,
                "npm": 10,
                "openjdk-17-jdk": 300,
                "openjdk-11-jdk": 250,
                "mysql-client": 25,
                "postgresql-client": 15,
                "redis-tools": 5,
            },
            "cache_settings": {
                "max_size": 1000,
                "ttl_seconds": 3600,
            },
            "layer_estimation": {
                "base_layer_mb": 8,
                "package_layer_mb": 12,
                "copy_layer_mb": 15,
                "run_layer_mb": 5,
            },
            "default_fallbacks": {
                "unknown_image_size_mb": 100,
                "unknown_package_size_mb": 10,
            }
        }

    def _find_user_config(self) -> Optional[str]:
        """Find user configuration file in home directory."""
        home = Path.home()
        config_files = [
            home / ".docker-optimizer.yml",
            home / ".docker-optimizer.yaml",
            home / ".docker-optimizer.json"
        ]

        for config_file in config_files:
            if config_file.exists():
                return str(config_file)

        return None

    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dict containing configuration

        Raises:
            ConfigError: If file cannot be parsed
        """
        path = Path(config_path)

        try:
            content = path.read_text()

            if path.suffix.lower() in ['.yml', '.yaml']:
                result = yaml.safe_load(content)
                return result if result is not None else {}
            elif path.suffix.lower() == '.json':
                return dict(json.loads(content))
            else:
                # Try YAML first, then JSON
                try:
                    result = yaml.safe_load(content)
                    return result if result is not None else {}
                except yaml.YAMLError:
                    return dict(json.loads(content))

        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigError(f"Failed to parse configuration file {config_path}: {e}") from e
        except Exception as e:
            raise ConfigError(f"Failed to read configuration file {config_path}: {e}") from e

    def _merge_config(self, user_config: Dict[str, Any]) -> None:
        """Merge user configuration with defaults.

        Args:
            user_config: User configuration to merge
        """
        for key, value in user_config.items():
            if key in self._config and isinstance(self._config[key], dict) and isinstance(value, dict):
                # Deep merge for dictionaries
                self._config[key].update(value)
            else:
                # Direct override for non-dict values
                self._config[key] = value

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides with comprehensive validation."""
        env_mappings = {
            # Cache settings
            'DOCKER_OPTIMIZER_CACHE_MAX_SIZE': ('cache_settings', 'max_size', int, lambda x: x > 0, "Must be a positive integer"),
            'DOCKER_OPTIMIZER_CACHE_TTL_SECONDS': ('cache_settings', 'ttl_seconds', int, lambda x: x > 0, "Must be a positive integer"),

            # Fallback settings
            'DOCKER_OPTIMIZER_UNKNOWN_IMAGE_SIZE': ('default_fallbacks', 'unknown_image_size_mb', int, lambda x: x > 0, "Must be a positive integer"),
            'DOCKER_OPTIMIZER_UNKNOWN_PACKAGE_SIZE': ('default_fallbacks', 'unknown_package_size_mb', int, lambda x: x > 0, "Must be a positive integer"),

            # Layer estimation
            'DOCKER_OPTIMIZER_BASE_LAYER_MB': ('layer_estimation', 'base_layer_mb', int, lambda x: x > 0, "Must be a positive integer"),
            'DOCKER_OPTIMIZER_PACKAGE_LAYER_MB': ('layer_estimation', 'package_layer_mb', int, lambda x: x > 0, "Must be a positive integer"),
            'DOCKER_OPTIMIZER_COPY_LAYER_MB': ('layer_estimation', 'copy_layer_mb', int, lambda x: x > 0, "Must be a positive integer"),
            'DOCKER_OPTIMIZER_RUN_LAYER_MB': ('layer_estimation', 'run_layer_mb', int, lambda x: x > 0, "Must be a positive integer"),

            # CLI behavior settings
            'DOCKER_OPTIMIZER_VERBOSE': ('cli_defaults', 'verbose', bool, None, None),
            'DOCKER_OPTIMIZER_OUTPUT_FORMAT': ('cli_defaults', 'output_format', str, lambda x: x in ['text', 'json', 'yaml'], "Must be one of: text, json, yaml"),
            'DOCKER_OPTIMIZER_SECURITY_SCAN': ('cli_defaults', 'security_scan', bool, None, None),
        }

        # Ensure cli_defaults section exists
        if 'cli_defaults' not in self._config:
            self._config['cli_defaults'] = {
                'verbose': False,
                'output_format': 'text',
                'security_scan': False
            }

        for env_var, (section, key, type_func, validator, error_msg) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Type conversion
                    if type_func == bool:
                        parsed_value = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        parsed_value = type_func(value)

                    # Validation
                    if validator and not validator(parsed_value):
                        raise ConfigError(
                            f"Invalid value for {env_var}: {error_msg}",
                            field_path=f"{section}.{key}",
                            suggestions=[f"Example: {env_var}=100" if type_func == int else f"Example: {env_var}=true"]
                        )

                    self._config[section][key] = parsed_value
                    logger.debug("Applied environment override: %s=%s", env_var, value)

                except (ValueError, TypeError) as e:
                    error_msg = f"Invalid environment variable {env_var}={value}: {e}"
                    suggestions = [
                        f"Check that the value is a valid {type_func.__name__}",
                        f"Example: {env_var}={'true' if type_func == bool else '100' if type_func == int else 'text'}"
                    ]
                    raise ConfigError(error_msg, field_path=f"{section}.{key}", suggestions=suggestions)

    def get_base_image_sizes(self) -> Dict[str, int]:
        """Get base image sizes configuration."""
        return dict(self._config["base_image_sizes"])

    def get_package_sizes(self) -> Dict[str, int]:
        """Get package sizes configuration."""
        return dict(self._config["package_sizes"])

    def get_cache_settings(self) -> Dict[str, int]:
        """Get cache settings configuration."""
        return dict(self._config["cache_settings"])

    def get_layer_estimation_settings(self) -> Dict[str, int]:
        """Get layer estimation settings."""
        return dict(self._config["layer_estimation"])

    def get_image_size(self, image_name: str) -> int:
        """Get size for specific image with fallback.

        Args:
            image_name: Docker image name

        Returns:
            Image size in MB
        """
        sizes = self.get_base_image_sizes()
        fallback: int = self._config["default_fallbacks"]["unknown_image_size_mb"]
        return sizes.get(image_name, fallback)

    def get_package_size(self, package_name: str) -> int:
        """Get size for specific package with fallback.

        Args:
            package_name: Package name

        Returns:
            Package size in MB
        """
        sizes = self.get_package_sizes()
        fallback: int = self._config["default_fallbacks"]["unknown_package_size_mb"]
        return sizes.get(package_name, fallback)

    def get_config_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()

    def validate_config(self) -> List[str]:
        """Validate current configuration and return list of issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate cache settings
        cache = self._config.get('cache_settings', {})
        if not isinstance(cache.get('max_size'), int) or cache.get('max_size', 0) <= 0:
            errors.append("cache_settings.max_size must be a positive integer")
        if not isinstance(cache.get('ttl_seconds'), int) or cache.get('ttl_seconds', 0) <= 0:
            errors.append("cache_settings.ttl_seconds must be a positive integer")

        # Validate layer estimation
        layer = self._config.get('layer_estimation', {})
        for key in ['base_layer_mb', 'package_layer_mb', 'copy_layer_mb', 'run_layer_mb']:
            if not isinstance(layer.get(key), int) or layer.get(key, 0) <= 0:
                errors.append(f"layer_estimation.{key} must be a positive integer")

        # Validate fallbacks
        fallbacks = self._config.get('default_fallbacks', {})
        for key in ['unknown_image_size_mb', 'unknown_package_size_mb']:
            if not isinstance(fallbacks.get(key), int) or fallbacks.get(key, 0) <= 0:
                errors.append(f"default_fallbacks.{key} must be a positive integer")

        # Validate base image sizes
        base_images = self._config.get('base_image_sizes', {})
        if not isinstance(base_images, dict):
            errors.append("base_image_sizes must be a dictionary")
        else:
            for image, size in base_images.items():
                if not isinstance(size, int) or size <= 0:
                    errors.append(f"base_image_sizes.{image} must be a positive integer")

        # Validate package sizes
        packages = self._config.get('package_sizes', {})
        if not isinstance(packages, dict):
            errors.append("package_sizes must be a dictionary")
        else:
            for package, size in packages.items():
                if not isinstance(size, int) or size <= 0:
                    errors.append(f"package_sizes.{package} must be a positive integer")

        return errors

    def get_cli_defaults(self) -> Dict[str, Any]:
        """Get CLI default settings.
        
        Returns:
            Dictionary of CLI default values
        """
        return self._config.get('cli_defaults', {
            'verbose': False,
            'output_format': 'text',
            'security_scan': False
        })

    def get_supported_env_vars(self) -> Dict[str, str]:
        """Get list of supported environment variables with descriptions.
        
        Returns:
            Dictionary mapping env var names to descriptions
        """
        return {
            'DOCKER_OPTIMIZER_CACHE_MAX_SIZE': 'Maximum cache size (positive integer)',
            'DOCKER_OPTIMIZER_CACHE_TTL_SECONDS': 'Cache TTL in seconds (positive integer)',
            'DOCKER_OPTIMIZER_UNKNOWN_IMAGE_SIZE': 'Fallback size for unknown images in MB (positive integer)',
            'DOCKER_OPTIMIZER_UNKNOWN_PACKAGE_SIZE': 'Fallback size for unknown packages in MB (positive integer)',
            'DOCKER_OPTIMIZER_BASE_LAYER_MB': 'Base layer size estimate in MB (positive integer)',
            'DOCKER_OPTIMIZER_PACKAGE_LAYER_MB': 'Package layer size estimate in MB (positive integer)',
            'DOCKER_OPTIMIZER_COPY_LAYER_MB': 'Copy layer size estimate in MB (positive integer)',
            'DOCKER_OPTIMIZER_RUN_LAYER_MB': 'Run layer size estimate in MB (positive integer)',
            'DOCKER_OPTIMIZER_VERBOSE': 'Enable verbose output (true/false)',
            'DOCKER_OPTIMIZER_OUTPUT_FORMAT': 'Default output format (text/json/yaml)',
            'DOCKER_OPTIMIZER_SECURITY_SCAN': 'Enable security scanning by default (true/false)',
        }

    def save_to_file(self, config_path: str, format: str = "yaml") -> None:
        """Save current configuration to file with validation.

        Args:
            config_path: Path where to save configuration
            format: File format ('yaml' or 'json')
            
        Raises:
            ConfigError: If configuration is invalid or cannot be saved
        """
        # Validate configuration before saving
        validation_errors = self.validate_config()
        if validation_errors:
            raise ConfigError(
                "Cannot save invalid configuration",
                suggestions=["Fix the following validation errors:"] + validation_errors
            )

        path = Path(config_path)

        try:
            if format.lower() == "json":
                content = json.dumps(self._config, indent=2, sort_keys=True)
            else:
                content = yaml.dump(self._config, default_flow_style=False, indent=2, sort_keys=True)

            # Create parent directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            logger.info("Configuration saved to %s", config_path)

        except Exception as e:
            raise ConfigError(
                f"Failed to save configuration to {config_path}: {e}",
                suggestions=[
                    "Check that the directory exists and is writable",
                    "Ensure you have permission to write to the target directory"
                ]
            ) from e
