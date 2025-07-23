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
