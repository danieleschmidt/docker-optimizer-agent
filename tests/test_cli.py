"""Tests for quantum task planner CLI."""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from click.testing import CliRunner
from unittest.mock import Mock, patch

from src.quantum_task_planner.cli import cli, generate_example_schedule


class TestCLI:
    """Test CLI functionality."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "Quantum-Inspired Task Planner" in result.output
    
    def test_cli_version(self, runner):
        """Test CLI version display."""
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
    
    def test_generate_example_command(self, runner):
        """Test generate-example command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "example.json"
            
            result = runner.invoke(cli, [
                'generate-example',
                '--output', str(output_file),
                '--tasks', '5',
                '--resources', '3',
                '--complexity', 'medium'
            ])
            
            assert result.exit_code == 0
            assert output_file.exists()
            
            # Validate generated content
            with open(output_file) as f:
                data = json.load(f)
                assert "id" in data
                assert "tasks" in data
                assert "resources" in data
                assert len(data["tasks"]) == 5
                assert len(data["resources"]) == 3


class TestGenerateExample:
    """Test example generation functionality."""
    
    def test_generate_example_schedule_simple(self):
        """Test simple example generation."""
        data = generate_example_schedule(3, 2, "simple")
        
        assert "id" in data
        assert "name" in data
        assert "tasks" in data
        assert "resources" in data
        assert len(data["tasks"]) == 3
        assert len(data["resources"]) == 2
    
    def test_generate_example_schedule_complex(self):
        """Test complex example generation."""
        data = generate_example_schedule(4, 2, "complex")
        
        assert len(data["tasks"]) == 4
        assert len(data["resources"]) == 2
        
        # Validate task structure
        for task in data["tasks"]:
            assert "id" in task
            assert "name" in task
            assert "duration_seconds" in task
            assert "priority" in task
            assert task["duration_seconds"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])