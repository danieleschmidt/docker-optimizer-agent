"""Configuration management for Docker Optimizer Agent."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration-related errors."""
    pass


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
            except Exception as e:
                logger.warning("Failed to load configuration from %s: %s", config_path, e)
        
        # Apply environment variable overrides
        self._apply_env_overrides()

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
            raise ConfigError(f"Failed to parse configuration file {config_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to read configuration file {config_path}: {e}")

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
        """Apply environment variable overrides."""
        env_mappings = {
            'DOCKER_OPTIMIZER_CACHE_MAX_SIZE': ('cache_settings', 'max_size', int),
            'DOCKER_OPTIMIZER_CACHE_TTL_SECONDS': ('cache_settings', 'ttl_seconds', int),
            'DOCKER_OPTIMIZER_UNKNOWN_IMAGE_SIZE': ('default_fallbacks', 'unknown_image_size_mb', int),
            'DOCKER_OPTIMIZER_UNKNOWN_PACKAGE_SIZE': ('default_fallbacks', 'unknown_package_size_mb', int),
        }
        
        for env_var, (section, key, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    self._config[section][key] = type_func(value)
                    logger.debug("Applied environment override: %s=%s", env_var, value)
                except (ValueError, TypeError) as e:
                    logger.warning("Invalid environment variable %s=%s: %s", env_var, value, e)

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

    def save_to_file(self, config_path: str, format: str = "yaml") -> None:
        """Save current configuration to file.
        
        Args:
            config_path: Path where to save configuration
            format: File format ('yaml' or 'json')
        """
        path = Path(config_path)
        
        try:
            if format.lower() == "json":
                content = json.dumps(self._config, indent=2)
            else:
                content = yaml.dump(self._config, default_flow_style=False, indent=2)
            
            path.write_text(content)
            logger.info("Configuration saved to %s", config_path)
            
        except Exception as e:
            raise ConfigError(f"Failed to save configuration to {config_path}: {e}")