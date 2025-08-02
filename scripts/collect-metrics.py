#!/usr/bin/env python3
"""
Automated metrics collection script for Docker Optimizer Agent.

This script collects various metrics about the project including:
- Code quality metrics
- Performance metrics
- Usage statistics
- Security metrics
- Development metrics
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import requests


class MetricsCollector:
    """Collect and aggregate project metrics."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.metrics = {}
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "github_repo": "danieleschmidt/docker-optimizer-agent",
            "pypi_package": "docker-optimizer-agent",
            "docker_image": "docker-optimizer",
            "output_file": ".github/project-metrics.json",
            "enable_github_api": True,
            "enable_pypi_stats": True,
            "enable_docker_stats": True,
        }
        
        if config_path and config_path.exists():
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def run_command(self, command: list[str]) -> tuple[int, str, str]:
        """Run a shell command and return result."""
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics."""
        print("📊 Collecting Git metrics...")
        
        metrics = {}
        
        # Get commit count
        code, stdout, _ = self.run_command(["git", "rev-list", "--count", "HEAD"])
        if code == 0:
            metrics["total_commits"] = int(stdout.strip())
        
        # Get contributors count
        code, stdout, _ = self.run_command([
            "git", "shortlog", "-sn", "--all", "--no-merges"
        ])
        if code == 0:
            metrics["total_contributors"] = len(stdout.strip().split('\n'))
        
        # Get recent commit activity
        code, stdout, _ = self.run_command([
            "git", "log", "--since=1.month.ago", "--oneline"
        ])
        if code == 0:
            metrics["commits_last_month"] = len(stdout.strip().split('\n')) if stdout.strip() else 0
        
        # Get current branch
        code, stdout, _ = self.run_command(["git", "branch", "--show-current"])
        if code == 0:
            metrics["current_branch"] = stdout.strip()
            
        return metrics
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        print("🔍 Collecting code quality metrics...")
        
        metrics = {}
        
        # Run coverage if available
        code, stdout, _ = self.run_command([
            "python", "-m", "pytest", "--cov=docker_optimizer", 
            "--cov-report=json", "--cov-report=term-missing", 
            "-q", "tests/"
        ])
        
        if code == 0:
            try:
                with open("coverage.json") as f:
                    cov_data = json.load(f)
                    metrics["code_coverage"] = round(cov_data["totals"]["percent_covered"], 2)
            except FileNotFoundError:
                metrics["code_coverage"] = None
        
        # Count lines of code
        code, stdout, _ = self.run_command([
            "find", "src/", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"
        ])
        if code == 0:
            lines = stdout.strip().split('\n')
            if lines and 'total' in lines[-1]:
                metrics["lines_of_code"] = int(lines[-1].split()[0])
        
        # Run linting
        code, stdout, stderr = self.run_command([
            "ruff", "check", "src/", "--format", "json"
        ])
        if code == 0 or stderr:  # ruff returns non-zero on issues
            try:
                issues = json.loads(stderr) if stderr else []
                metrics["linting_issues"] = len(issues)
            except json.JSONDecodeError:
                metrics["linting_issues"] = None
        
        # Count test files
        test_files = list(Path("tests/").glob("test_*.py"))
        metrics["test_files"] = len(test_files)
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        print("🔒 Collecting security metrics...")
        
        metrics = {}
        
        # Run bandit security scan
        code, stdout, stderr = self.run_command([
            "bandit", "-r", "src/", "-f", "json"
        ])
        
        if code == 0 or stdout:  # bandit may return non-zero on issues
            try:
                bandit_data = json.loads(stdout)
                metrics["security_issues"] = {
                    "high": len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "HIGH"]),
                    "medium": len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "MEDIUM"]),
                    "low": len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "LOW"])
                }
            except json.JSONDecodeError:
                pass
        
        # Check dependency vulnerabilities
        code, stdout, stderr = self.run_command([
            "safety", "check", "--json"
        ])
        
        if code == 0:
            try:
                safety_data = json.loads(stdout)
                metrics["dependency_vulnerabilities"] = len(safety_data)
            except json.JSONDecodeError:
                pass
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics."""
        print("⚡ Collecting performance metrics...")
        
        metrics = {}
        
        # Docker image sizes
        images = ["docker-optimizer:dev", "docker-optimizer:latest", "docker-optimizer:cli"]
        for image in images:
            code, stdout, _ = self.run_command([
                "docker", "image", "inspect", image, "--format", "{{.Size}}"
            ])
            if code == 0:
                size_bytes = int(stdout.strip())
                size_mb = round(size_bytes / (1024 * 1024), 1)
                metrics[f"{image.split(':')[1]}_image_size_mb"] = size_mb
        
        # Python package size
        if Path("dist/").exists():
            wheel_files = list(Path("dist/").glob("*.whl"))
            if wheel_files:
                size_bytes = wheel_files[0].stat().st_size
                metrics["package_size_kb"] = round(size_bytes / 1024, 1)
        
        return metrics
    
    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect GitHub repository metrics via API."""
        if not self.config.get("enable_github_api"):
            return {}
            
        print("🐙 Collecting GitHub metrics...")
        
        repo = self.config["github_repo"]
        api_url = f"https://api.github.com/repos/{repo}"
        
        try:
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "github_stars": data.get("stargazers_count", 0),
                    "github_forks": data.get("forks_count", 0),
                    "github_watchers": data.get("watchers_count", 0),
                    "github_issues": data.get("open_issues_count", 0),
                    "github_size_kb": data.get("size", 0),
                    "github_language": data.get("language", "Unknown"),
                    "github_created_at": data.get("created_at"),
                    "github_updated_at": data.get("updated_at"),
                }
        except requests.RequestException as e:
            print(f"Warning: Could not fetch GitHub metrics: {e}")
            
        return {}
    
    def collect_pypi_metrics(self) -> Dict[str, Any]:
        """Collect PyPI package metrics."""
        if not self.config.get("enable_pypi_stats"):
            return {}
            
        print("📦 Collecting PyPI metrics...")
        
        package = self.config["pypi_package"]
        api_url = f"https://pypi.org/pypi/{package}/json"
        
        try:
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                info = data.get("info", {})
                
                # Get download stats (requires separate API)
                downloads_url = f"https://pypistats.org/api/packages/{package}/recent"
                downloads_response = requests.get(downloads_url, timeout=10)
                downloads_data = downloads_response.json() if downloads_response.status_code == 200 else {}
                
                return {
                    "pypi_version": info.get("version"),
                    "pypi_author": info.get("author"),
                    "pypi_license": info.get("license"),
                    "pypi_downloads_last_month": downloads_data.get("data", {}).get("last_month", 0),
                    "pypi_downloads_last_week": downloads_data.get("data", {}).get("last_week", 0),
                    "pypi_downloads_last_day": downloads_data.get("data", {}).get("last_day", 0),
                }
        except requests.RequestException as e:
            print(f"Warning: Could not fetch PyPI metrics: {e}")
            
        return {}
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        print("🚀 Starting metrics collection...")
        
        all_metrics = {
            "collection_info": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "collector_version": "1.0.0",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            }
        }
        
        # Collect different types of metrics
        collectors = [
            ("git", self.collect_git_metrics),
            ("code_quality", self.collect_code_quality_metrics),
            ("security", self.collect_security_metrics),
            ("performance", self.collect_performance_metrics),
            ("github", self.collect_github_metrics),
            ("pypi", self.collect_pypi_metrics),
        ]
        
        for name, collector_func in collectors:
            try:
                metrics = collector_func()
                all_metrics[name] = metrics
                print(f"✅ Collected {name} metrics: {len(metrics)} items")
            except Exception as e:
                print(f"❌ Failed to collect {name} metrics: {e}")
                all_metrics[name] = {"error": str(e)}
        
        return all_metrics
    
    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to output file."""
        output_file = Path(self.config["output_file"])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        
        print(f"📄 Metrics saved to {output_file}")
        
        # Also save a timestamped version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = output_file.parent / f"metrics_backup_{timestamp}.json"
        with open(backup_file, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        
        print(f"💾 Backup saved to {backup_file}")
    
    def generate_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate a human-readable summary of metrics."""
        summary = []
        summary.append("# Project Metrics Summary")
        summary.append(f"Generated on: {metrics['collection_info']['timestamp']}")
        summary.append("")
        
        # Git metrics
        git_metrics = metrics.get("git", {})
        if git_metrics:
            summary.append("## Repository Statistics")
            summary.append(f"- Total commits: {git_metrics.get('total_commits', 'N/A')}")
            summary.append(f"- Contributors: {git_metrics.get('total_contributors', 'N/A')}")
            summary.append(f"- Recent activity: {git_metrics.get('commits_last_month', 'N/A')} commits last month")
            summary.append("")
        
        # Code quality
        quality_metrics = metrics.get("code_quality", {})
        if quality_metrics:
            summary.append("## Code Quality")
            summary.append(f"- Code coverage: {quality_metrics.get('code_coverage', 'N/A')}%")
            summary.append(f"- Lines of code: {quality_metrics.get('lines_of_code', 'N/A')}")
            summary.append(f"- Test files: {quality_metrics.get('test_files', 'N/A')}")
            summary.append(f"- Linting issues: {quality_metrics.get('linting_issues', 'N/A')}")
            summary.append("")
        
        # Security
        security_metrics = metrics.get("security", {})
        if security_metrics and "security_issues" in security_metrics:
            issues = security_metrics["security_issues"]
            summary.append("## Security")
            summary.append(f"- High severity issues: {issues.get('high', 0)}")
            summary.append(f"- Medium severity issues: {issues.get('medium', 0)}")
            summary.append(f"- Low severity issues: {issues.get('low', 0)}")
            summary.append("")
        
        # GitHub stats
        github_metrics = metrics.get("github", {})
        if github_metrics:
            summary.append("## GitHub Statistics")
            summary.append(f"- Stars: {github_metrics.get('github_stars', 'N/A')}")
            summary.append(f"- Forks: {github_metrics.get('github_forks', 'N/A')}")
            summary.append(f"- Open issues: {github_metrics.get('github_issues', 'N/A')}")
            summary.append("")
        
        return "\n".join(summary)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument(
        "--config", 
        type=Path,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output", 
        type=Path,
        default=Path(".github/project-metrics.json"),
        help="Output file path"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate human-readable summary"
    )
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = MetricsCollector(args.config)
    if args.output:
        collector.config["output_file"] = str(args.output)
    
    # Collect metrics
    metrics = collector.collect_all_metrics()
    
    # Save metrics
    collector.save_metrics(metrics)
    
    # Generate summary if requested
    if args.summary:
        summary = collector.generate_summary(metrics)
        print("\n" + "="*50)
        print(summary)
        
        summary_file = Path(args.output).parent / "metrics_summary.md"
        with open(summary_file, "w") as f:
            f.write(summary)
        print(f"\n📋 Summary saved to {summary_file}")
    
    print(f"\n🎉 Metrics collection completed successfully!")


if __name__ == "__main__":
    main()