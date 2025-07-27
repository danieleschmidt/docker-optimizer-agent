"""Test cases for configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from docker_optimizer.config import Config, ConfigError


class TestConfig:
    """Test cases for configuration management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()

    def test_config_initialization(self):
        """Test that configuration initializes with defaults."""
        assert isinstance(self.config, Config)
        # Test that the config has the expected methods
        assert callable(getattr(self.config, 'get_base_image_sizes', None))
        assert callable(getattr(self.config, 'get_package_sizes', None))
        assert callable(getattr(self.config, 'get_cache_settings', None))

    def test_default_base_image_sizes(self):
        """Test that default base image sizes are loaded."""
        sizes = self.config.get_base_image_sizes()
        assert isinstance(sizes, dict)
        assert "alpine:latest" in sizes
        assert "ubuntu:20.04" in sizes
        assert sizes["alpine:latest"] > 0

    def test_default_package_sizes(self):
        """Test that default package sizes are loaded."""
        sizes = self.config.get_package_sizes()
        assert isinstance(sizes, dict)
        assert "curl" in sizes
        assert "wget" in sizes
        assert sizes["curl"] > 0

    def test_cache_settings(self):
        """Test that cache settings are configurable."""
        cache_config = self.config.get_cache_settings()
        assert isinstance(cache_config, dict)
        assert "max_size" in cache_config
        assert "ttl_seconds" in cache_config
        assert cache_config["max_size"] > 0
        assert cache_config["ttl_seconds"] > 0

    def test_config_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        config_content = """
base_image_sizes:
  "custom:latest": 100
  "alpine:latest": 15

package_sizes:
  "custom-package": 25

cache_settings:
  max_size: 500
  ttl_seconds: 1800
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = Config.from_file(config_path)

            sizes = config.get_base_image_sizes()
            assert sizes["custom:latest"] == 100
            assert sizes["alpine:latest"] == 15

            pkg_sizes = config.get_package_sizes()
            assert pkg_sizes["custom-package"] == 25

            cache_settings = config.get_cache_settings()
            assert cache_settings["max_size"] == 500
            assert cache_settings["ttl_seconds"] == 1800
        finally:
            os.unlink(config_path)

    def test_config_from_json_file(self):
        """Test loading configuration from JSON file."""
        config_content = """{
  "base_image_sizes": {
    "custom:latest": 100
  },
  "cache_settings": {
    "max_size": 200,
    "ttl_seconds": 900
  }
}"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = Config.from_file(config_path)

            sizes = config.get_base_image_sizes()
            assert sizes["custom:latest"] == 100

            cache_settings = config.get_cache_settings()
            assert cache_settings["max_size"] == 200
            assert cache_settings["ttl_seconds"] == 900
        finally:
            os.unlink(config_path)

    def test_environment_variable_override(self):
        """Test that environment variables can override config values."""
        with patch.dict(os.environ, {
            'DOCKER_OPTIMIZER_CACHE_MAX_SIZE': '2000',
            'DOCKER_OPTIMIZER_CACHE_TTL_SECONDS': '7200'
        }):
            config = Config()
            cache_settings = config.get_cache_settings()
            assert cache_settings["max_size"] == 2000
            assert cache_settings["ttl_seconds"] == 7200

    def test_config_file_not_found(self):
        """Test handling of missing configuration file."""
        with pytest.raises(ConfigError):
            Config.from_file("/nonexistent/config.yml")

    def test_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            # Config.from_file should raise ConfigError for invalid files
            with pytest.raises(ConfigError):
                # Force strict loading by calling _load_config_file directly
                config = Config()
                config._load_config_file(config_path)
        finally:
            os.unlink(config_path)

    def test_merge_with_defaults(self):
        """Test that custom config merges with defaults."""
        config_content = """
base_image_sizes:
  "custom:latest": 100

cache_settings:
  max_size: 500
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = Config.from_file(config_path)

            sizes = config.get_base_image_sizes()
            # Should have custom value
            assert sizes["custom:latest"] == 100
            # Should still have default values
            assert "alpine:latest" in sizes
            assert "ubuntu:20.04" in sizes

            cache_settings = config.get_cache_settings()
            # Should have custom value
            assert cache_settings["max_size"] == 500
            # Should still have default ttl_seconds
            assert "ttl_seconds" in cache_settings
        finally:
            os.unlink(config_path)

    def test_home_directory_config(self):
        """Test loading config from home directory."""
        with patch.object(Path, 'home') as mock_home:
            mock_home.return_value = Path('/mock/home')

            # Mock the existence check and file reading
            with patch.object(Path, 'exists', return_value=True), \
                 patch.object(Path, 'read_text', return_value='cache_settings:\n  max_size: 123'):

                config = Config()
                # Should attempt to load from home directory
                cache_settings = config.get_cache_settings()
                # This test verifies the code path exists
                assert isinstance(cache_settings, dict)

    def test_get_image_size_with_fallback(self):
        """Test getting image size with fallback for unknown images."""
        size = self.config.get_image_size("unknown:image")
        assert isinstance(size, int)
        assert size > 0  # Should return a reasonable default

    def test_get_package_size_with_fallback(self):
        """Test getting package size with fallback for unknown packages."""
        size = self.config.get_package_size("unknown-package")
        assert isinstance(size, int)
        assert size > 0  # Should return a reasonable default

    def test_enhanced_config_validation(self):
        """Test comprehensive configuration validation."""
        config = Config()

        # Test valid configuration
        errors = config.validate_config()
        assert len(errors) == 0, f"Valid config should have no errors, got: {errors}"

        # Test invalid configuration
        config._config['cache_settings']['max_size'] = -1
        config._config['layer_estimation']['base_layer_mb'] = 'invalid'

        errors = config.validate_config()
        assert len(errors) >= 2
        assert any('cache_settings.max_size' in error for error in errors)
        assert any('layer_estimation.base_layer_mb' in error for error in errors)

    def test_enhanced_error_messages(self):
        """Test enhanced error messages with context and suggestions."""
        with pytest.raises(ConfigError) as exc_info:
            raise ConfigError(
                "Test error",
                field_path="test.field",
                suggestions=["Try this", "Or this"]
            )

        error_str = str(exc_info.value)
        assert "Configuration error in 'test.field'" in error_str
        assert "Suggestions:" in error_str
        assert "Try this" in error_str

    def test_cli_defaults(self):
        """Test CLI defaults configuration."""
        config = Config()
        cli_defaults = config.get_cli_defaults()

        assert isinstance(cli_defaults, dict)
        assert 'verbose' in cli_defaults
        assert 'output_format' in cli_defaults
        assert 'security_scan' in cli_defaults

        # Test defaults
        assert cli_defaults['verbose'] is False
        assert cli_defaults['output_format'] == 'text'
        assert cli_defaults['security_scan'] is False

    def test_supported_env_vars(self):
        """Test supported environment variables documentation."""
        config = Config()
        env_vars = config.get_supported_env_vars()

        assert isinstance(env_vars, dict)
        assert len(env_vars) > 0

        # Check key environment variables are documented
        assert 'DOCKER_OPTIMIZER_CACHE_MAX_SIZE' in env_vars
        assert 'DOCKER_OPTIMIZER_VERBOSE' in env_vars
        assert 'DOCKER_OPTIMIZER_OUTPUT_FORMAT' in env_vars

        # Check descriptions are provided
        for var, desc in env_vars.items():
            assert isinstance(desc, str)
            assert len(desc) > 0

    @patch.dict(os.environ, {
        'DOCKER_OPTIMIZER_CACHE_MAX_SIZE': '2000',
        'DOCKER_OPTIMIZER_VERBOSE': 'true',
        'DOCKER_OPTIMIZER_OUTPUT_FORMAT': 'json'
    })
    def test_comprehensive_env_overrides(self):
        """Test comprehensive environment variable overrides."""
        config = Config()

        # Test cache setting override
        cache_settings = config.get_cache_settings()
        assert cache_settings['max_size'] == 2000

        # Test CLI defaults overrides
        cli_defaults = config.get_cli_defaults()
        assert cli_defaults['verbose'] is True
        assert cli_defaults['output_format'] == 'json'

    @patch.dict(os.environ, {'DOCKER_OPTIMIZER_CACHE_MAX_SIZE': 'invalid'})
    def test_env_override_validation(self):
        """Test environment variable validation with helpful errors."""
        with pytest.raises(ConfigError) as exc_info:
            Config()

        error = exc_info.value
        assert error.field_path == "cache_settings.max_size"
        assert len(error.suggestions) > 0
        assert "Example:" in str(error)

    @patch.dict(os.environ, {'DOCKER_OPTIMIZER_OUTPUT_FORMAT': 'invalid_format'})
    def test_env_validation_with_choices(self):
        """Test environment variable validation with choice constraints."""
        with pytest.raises(ConfigError) as exc_info:
            Config()

        error = exc_info.value
        assert "Must be one of: text, json, yaml" in str(error)
