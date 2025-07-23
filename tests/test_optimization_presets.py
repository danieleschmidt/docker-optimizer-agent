"""Tests for optimization presets and profiles functionality."""

import pytest
from pathlib import Path
import tempfile
import json
import yaml

from docker_optimizer.optimization_presets import (
    PresetManager,
    OptimizationPreset,
    PresetType,
    CustomPreset
)


class TestOptimizationPresets:
    """Test optimization presets and profiles functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preset_manager = PresetManager()

    def test_development_preset(self):
        """Test development optimization preset."""
        preset = self.preset_manager.get_preset(PresetType.DEVELOPMENT)
        
        assert isinstance(preset, OptimizationPreset)
        assert preset.name == "Development"
        assert preset.preset_type == PresetType.DEVELOPMENT.value
        assert "fast builds" in preset.description.lower()
        assert len(preset.optimizations) > 0
        
        # Development preset should prioritize build speed over size
        optimization_names = [opt.name.lower() for opt in preset.optimizations]
        assert any("cache" in name for name in optimization_names)
        assert any("layer" in name for name in optimization_names)

    def test_production_preset(self):
        """Test production optimization preset."""
        preset = self.preset_manager.get_preset(PresetType.PRODUCTION)
        
        assert isinstance(preset, OptimizationPreset)
        assert preset.name == "Production"
        assert preset.preset_type == PresetType.PRODUCTION.value
        assert "security" in preset.description.lower() or "size" in preset.description.lower()
        assert len(preset.optimizations) > 0
        
        # Production preset should prioritize security and size
        optimization_names = [opt.name.lower() for opt in preset.optimizations]
        assert any("security" in name for name in optimization_names)
        assert any("multi-stage" in name or "distroless" in name for name in optimization_names)

    def test_web_app_preset(self):
        """Test web application industry preset."""
        preset = self.preset_manager.get_preset(PresetType.WEB_APP)
        
        assert isinstance(preset, OptimizationPreset)
        assert preset.name == "Web Application"
        assert preset.preset_type == PresetType.WEB_APP.value
        assert len(preset.optimizations) > 0
        
        # Web app preset should include web-specific optimizations
        optimization_names = [opt.name.lower() for opt in preset.optimizations]
        assert any("static" in name or "nginx" in name for name in optimization_names)

    def test_ml_preset(self):
        """Test machine learning industry preset."""
        preset = self.preset_manager.get_preset(PresetType.ML)
        
        assert isinstance(preset, OptimizationPreset)
        assert preset.name == "Machine Learning"
        assert preset.preset_type == PresetType.ML.value
        assert len(preset.optimizations) > 0
        
        # ML preset should include ML-specific optimizations
        optimization_names = [opt.name.lower() for opt in preset.optimizations]
        assert any("gpu" in name or "cuda" in name or "python" in name for name in optimization_names)

    def test_data_processing_preset(self):
        """Test data processing industry preset."""
        preset = self.preset_manager.get_preset(PresetType.DATA_PROCESSING)
        
        assert isinstance(preset, OptimizationPreset)
        assert preset.name == "Data Processing"
        assert preset.preset_type == PresetType.DATA_PROCESSING.value
        assert len(preset.optimizations) > 0

    def test_list_available_presets(self):
        """Test listing all available presets."""
        presets = self.preset_manager.list_presets()
        
        assert isinstance(presets, list)
        assert len(presets) >= 5  # At least dev, prod, web, ml, data processing
        
        preset_types = [preset.preset_type for preset in presets]
        assert PresetType.DEVELOPMENT.value in preset_types
        assert PresetType.PRODUCTION.value in preset_types
        assert PresetType.WEB_APP.value in preset_types
        assert PresetType.ML.value in preset_types
        assert PresetType.DATA_PROCESSING.value in preset_types

    def test_apply_preset_to_dockerfile(self):
        """Test applying a preset to a Dockerfile."""
        dockerfile_content = """
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
"""
        
        preset = self.preset_manager.get_preset(PresetType.PRODUCTION)
        optimized_dockerfile = self.preset_manager.apply_preset(dockerfile_content, preset)
        
        assert isinstance(optimized_dockerfile, str)
        assert len(optimized_dockerfile) > len(dockerfile_content)
        # Should contain optimizations
        assert "slim" in optimized_dockerfile.lower() or "alpine" in optimized_dockerfile.lower()

    def test_create_custom_preset(self):
        """Test creating a custom optimization preset."""
        custom_preset = self.preset_manager.create_custom_preset(
            name="My Custom Preset",
            description="Custom preset for my use case",
            base_preset=PresetType.PRODUCTION,
            additional_optimizations=["Use specific base image", "Add health checks"]
        )
        
        assert isinstance(custom_preset, CustomPreset)
        assert custom_preset.name == "My Custom Preset"
        assert custom_preset.base_preset == PresetType.PRODUCTION.value
        assert len(custom_preset.additional_optimizations) == 2

    def test_save_custom_preset(self):
        """Test saving a custom preset to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            preset_file = Path(temp_dir) / "custom_preset.json"
            
            custom_preset = self.preset_manager.create_custom_preset(
                name="Test Preset",
                description="Test description",
                base_preset=PresetType.DEVELOPMENT
            )
            
            self.preset_manager.save_custom_preset(custom_preset, preset_file)
            
            assert preset_file.exists()
            
            # Verify file content
            with open(preset_file) as f:
                data = json.load(f)
            
            assert data["name"] == "Test Preset"
            assert data["base_preset"] == "DEVELOPMENT"

    def test_load_custom_preset(self):
        """Test loading a custom preset from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            preset_file = Path(temp_dir) / "custom_preset.yaml"
            
            # Create a custom preset file
            preset_data = {
                "name": "Loaded Preset",
                "description": "Loaded from file",
                "base_preset": "PRODUCTION",
                "additional_optimizations": ["Custom optimization 1", "Custom optimization 2"]
            }
            
            with open(preset_file, 'w') as f:
                yaml.dump(preset_data, f)
            
            loaded_preset = self.preset_manager.load_custom_preset(preset_file)
            
            assert isinstance(loaded_preset, CustomPreset)
            assert loaded_preset.name == "Loaded Preset"
            assert loaded_preset.base_preset == PresetType.PRODUCTION.value
            assert len(loaded_preset.additional_optimizations) == 2

    def test_preset_comparison(self):
        """Test comparing different presets."""
        dev_preset = self.preset_manager.get_preset(PresetType.DEVELOPMENT)
        prod_preset = self.preset_manager.get_preset(PresetType.PRODUCTION)
        
        comparison = self.preset_manager.compare_presets(dev_preset, prod_preset)
        
        assert isinstance(comparison, dict)
        assert "development_only" in comparison
        assert "production_only" in comparison
        assert "common" in comparison

    def test_preset_recommendations(self):
        """Test getting preset recommendations based on project type."""
        # Test web application recommendation
        recommendations = self.preset_manager.get_preset_recommendations(
            project_type="web",
            deployment_target="cloud",
            performance_priority="balanced"
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any(rec.preset_type == PresetType.WEB_APP.value for rec in recommendations)

    def test_preset_validation(self):
        """Test preset validation functionality."""
        preset = self.preset_manager.get_preset(PresetType.PRODUCTION)
        
        validation_result = self.preset_manager.validate_preset(preset)
        
        assert isinstance(validation_result, dict)
        assert "is_valid" in validation_result
        assert "issues" in validation_result
        assert validation_result["is_valid"] is True

    def test_merge_presets(self):
        """Test merging multiple presets."""
        base_preset = self.preset_manager.get_preset(PresetType.PRODUCTION)
        addon_preset = self.preset_manager.get_preset(PresetType.WEB_APP)
        
        merged_preset = self.preset_manager.merge_presets([base_preset, addon_preset])
        
        assert isinstance(merged_preset, OptimizationPreset)
        assert len(merged_preset.optimizations) >= len(base_preset.optimizations)
        assert "Merged" in merged_preset.name