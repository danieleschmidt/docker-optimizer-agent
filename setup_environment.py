#!/usr/bin/env python3
"""
Environment setup script for Docker Optimizer Agent development.
Ensures all dependencies are installed and development environment is ready.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, check=True, timeout=300):
    """Run a shell command with proper error handling."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=check, capture_output=True, 
            text=True, timeout=timeout
        )
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ {description} error: {str(e)}")
        return False


def main():
    """Set up development environment."""
    print("🚀 Setting up Docker Optimizer Agent development environment...")
    
    repo_path = Path(__file__).parent.absolute()
    print(f"📁 Working directory: {repo_path}")
    
    # Ensure we're in the right directory
    if not (repo_path / "pyproject.toml").exists():
        print("❌ pyproject.toml not found. Are you in the right directory?")
        sys.exit(1)
    
    success_count = 0
    total_steps = 7
    
    # Step 1: Upgrade pip
    if run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        success_count += 1
    
    # Step 2: Install package in development mode
    if run_command("pip install -e .", "Installing package in development mode"):
        success_count += 1
    
    # Step 3: Install development dependencies
    if run_command("pip install -e .[dev]", "Installing development dependencies"):
        success_count += 1
    
    # Step 4: Install security dependencies (optional)
    if run_command("pip install -e .[security]", "Installing security dependencies", check=False):
        success_count += 1
    
    # Step 5: Install pre-commit hooks
    if run_command("pre-commit install", "Installing pre-commit hooks", check=False):
        success_count += 1
    
    # Step 6: Create necessary directories
    directories = [
        ".terragon",
        "docs/status", 
        "benchmarks/results",
        "monitoring/data"
    ]
    
    for directory in directories:
        dir_path = repo_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")
    success_count += 1
    
    # Step 7: Verify installation
    verification_commands = [
        ("python -c 'import docker_optimizer; print(\"✅ Package import successful\")'", "Package import"),
        ("ruff --version", "Ruff linter"),
        ("black --version", "Black formatter"),
        ("pytest --version", "Pytest test runner"),
        ("mypy --version", "MyPy type checker")
    ]
    
    verification_success = 0
    for cmd, name in verification_commands:
        if run_command(cmd, f"Verifying {name}", check=False):
            verification_success += 1
    
    if verification_success >= 4:  # Allow one failure
        print("✅ Environment verification completed")
        success_count += 1
    
    # Summary
    print(f"\n📊 Environment Setup Summary:")
    print(f"✅ Completed: {success_count}/{total_steps} steps")
    
    if success_count >= 6:
        print("🎉 Development environment is ready!")
        print("\n🚀 Next steps:")
        print("  make test     # Run test suite")
        print("  make lint     # Run linting")
        print("  make dev      # Full development checks")
        return True
    else:
        print("⚠️  Some setup steps failed. Manual intervention may be required.")
        print("💡 Try running individual commands manually:")
        print("  pip install -e .[dev,security]")
        print("  pre-commit install")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)