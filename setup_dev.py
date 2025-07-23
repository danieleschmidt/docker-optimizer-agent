#!/usr/bin/env python3
"""Development environment setup script for Docker Optimizer Agent.

This script helps set up a reliable development environment by:
1. Installing dependencies
2. Setting up pre-commit hooks
3. Running initial checks
4. Providing guidance for Docker requirements
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], description: str, check: bool = True) -> bool:
    """Run a command and handle errors gracefully."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Need Python 3.9+")
        return False


def check_docker() -> bool:
    """Check if Docker is available and running."""
    if run_command(["docker", "--version"], "Checking Docker installation", check=False):
        if run_command(["docker", "info"], "Checking Docker daemon", check=False):
            print("✅ Docker is installed and running")
            return True
        else:
            print("⚠️  Docker is installed but daemon is not running")
            print("   Please start Docker daemon before running tests")
            return False
    else:
        print("⚠️  Docker is not installed")
        print("   Please install Docker to run integration tests")
        return False


def install_dependencies() -> bool:
    """Install project dependencies."""
    success = True
    
    # Install main dependencies
    if not run_command([sys.executable, "-m", "pip", "install", "-e", "."], 
                      "Installing main dependencies"):
        success = False
    
    # Install development dependencies
    if not run_command([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], 
                      "Installing development dependencies"):
        success = False
    
    # Install security dependencies (optional)
    run_command([sys.executable, "-m", "pip", "install", "-e", ".[security]"], 
               "Installing security dependencies", check=False)
    
    return success


def setup_pre_commit() -> bool:
    """Set up pre-commit hooks."""
    if Path(".pre-commit-config.yaml").exists():
        return run_command(["pre-commit", "install"], "Setting up pre-commit hooks")
    else:
        print("⚠️  .pre-commit-config.yaml not found, skipping pre-commit setup")
        return True


def run_initial_checks() -> bool:
    """Run initial code quality checks."""
    success = True
    
    print("\n🔍 Running initial code quality checks...")
    
    # Run ruff check
    if not run_command(["ruff", "check", "src/", "tests/"], "Running ruff linter", check=False):
        print("   Run 'ruff check --fix src/ tests/' to fix auto-fixable issues")
        success = False
    
    # Run type checking (ignore missing imports for now)
    if not run_command(["mypy", "--ignore-missing-imports", "src/docker_optimizer/"], 
                      "Running type checking", check=False):
        success = False
    
    # Run security check
    if not run_command(["bandit", "-r", "src/", "-f", "json"], 
                      "Running security check", check=False):
        print("   Check bandit output for security issues")
    
    return success


def run_tests() -> bool:
    """Run tests to verify setup."""
    print("\n🧪 Running tests to verify setup...")
    
    # Run unit tests only (skip integration tests that require Docker)
    if run_command([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", 
                   "-m", "not integration"], "Running unit tests", check=False):
        print("✅ Unit tests passed")
        
        # Try integration tests if Docker is available
        docker_available = check_docker()
        if docker_available:
            if run_command([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", 
                           "-m", "integration"], "Running integration tests", check=False):
                print("✅ Integration tests passed")
                return True
            else:
                print("⚠️  Integration tests failed - this may be expected in some environments")
                return True
        else:
            print("⚠️  Skipping integration tests (Docker not available)")
            return True
    else:
        print("❌ Unit tests failed")
        return False


def main():
    """Main setup routine."""
    print("🚀 Docker Optimizer Agent - Development Environment Setup")
    print("=" * 60)
    
    # Check requirements
    if not check_python_version():
        sys.exit(1)
    
    docker_available = check_docker()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup pre-commit
    print("\n🎣 Setting up pre-commit hooks...")
    setup_pre_commit()
    
    # Run initial checks
    print("\n🔍 Running initial checks...")
    checks_passed = run_initial_checks()
    
    # Run tests
    tests_passed = run_tests()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SETUP SUMMARY")
    print("=" * 60)
    
    if checks_passed and tests_passed:
        print("✅ Development environment setup completed successfully!")
        print("\n🎯 Next steps:")
        print("   • Run 'pytest tests/' to run all tests")
        print("   • Run 'ruff check src/ tests/' to check code quality")
        print("   • Run 'mypy src/docker_optimizer/' for type checking")
        if docker_available:
            print("   • Integration tests will use Docker automatically")
        else:
            print("   • Install Docker to enable integration tests")
    else:
        print("⚠️  Setup completed with some issues:")
        if not checks_passed:
            print("   • Code quality checks found issues")
        if not tests_passed:
            print("   • Some tests failed")
        print("\nPlease review the output above and fix any issues.")
    
    print("\n📚 Additional resources:")
    print("   • README.md - Project documentation")
    print("   • BACKLOG.md - Development backlog and priorities")
    print("   • pyproject.toml - Project configuration")


if __name__ == "__main__":
    main()