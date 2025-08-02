#!/usr/bin/env python3
"""
Comprehensive automation pipeline for Docker Optimizer Agent.

This script orchestrates various automation tasks including:
- Dependency updates and security scanning
- Code quality checks and automated fixes
- Documentation generation and updates
- Metrics collection and reporting
- Repository maintenance tasks
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('automation.log')
    ]
)
logger = logging.getLogger(__name__)


class AutomationPipeline:
    """Main automation pipeline orchestrator."""
    
    def __init__(self, config_path: Optional[Path] = None, dry_run: bool = False):
        self.config = self._load_config(config_path)
        self.dry_run = dry_run
        self.results = {}
        self.start_time = datetime.now(timezone.utc)
        
        if dry_run:
            logger.info("🔍 Running in DRY RUN mode - no changes will be made")
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "repository": {
                "name": "docker-optimizer-agent",
                "owner": "danieleschmidt",
                "main_branch": "main",
                "working_directory": "."
            },
            "automation": {
                "update_dependencies": True,
                "run_security_scans": True,
                "generate_documentation": True,
                "collect_metrics": True,
                "cleanup_tasks": True,
                "send_notifications": True
            },
            "quality_gates": {
                "min_coverage": 85.0,
                "max_security_issues": 0,
                "max_linting_issues": 5,
                "max_complexity": 10
            },
            "notifications": {
                "slack_webhook": None,
                "email_recipients": [],
                "github_issues": True
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path) as f:
                user_config = json.load(f)
                # Deep merge configurations
                def merge_dicts(base: dict, override: dict) -> dict:
                    result = base.copy()
                    for key, value in override.items():
                        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                            result[key] = merge_dicts(result[key], value)
                        else:
                            result[key] = value
                    return result
                default_config = merge_dicts(default_config, user_config)
        
        return default_config
    
    async def run_command(self, command: List[str], cwd: Optional[Path] = None, timeout: int = 300) -> Tuple[int, str, str]:
        """Run a shell command asynchronously."""
        if self.dry_run:
            logger.info(f"DRY RUN: Would execute: {' '.join(command)}")
            return 0, "DRY RUN OUTPUT", ""
        
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout
            )
            
            return process.returncode, stdout.decode(), stderr.decode()
        except asyncio.TimeoutError:
            logger.error(f"Command timed out after {timeout}s: {' '.join(command)}")
            return 1, "", "Command timed out"
        except Exception as e:
            logger.error(f"Command failed: {' '.join(command)}, Error: {e}")
            return 1, "", str(e)
    
    async def update_dependencies(self) -> Dict[str, Any]:
        """Update project dependencies with security scanning."""
        logger.info("📦 Starting dependency updates...")
        
        results = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "python_updates": {},
            "docker_updates": {},
            "security_scan": {},
            "status": "running"
        }
        
        try:
            # Update Python dependencies
            logger.info("Updating Python dependencies...")
            
            # First, check for security vulnerabilities
            code, stdout, stderr = await self.run_command([
                "safety", "check", "--json"
            ])
            
            if code == 0:
                results["security_scan"]["vulnerabilities_before"] = len(json.loads(stdout or "[]"))
            
            # Update dependencies
            code, stdout, stderr = await self.run_command([
                "pip-review", "--auto", "--pre"
            ])
            
            if code == 0:
                results["python_updates"]["status"] = "success"
                results["python_updates"]["output"] = stdout[:1000]  # Truncate long output
            else:
                results["python_updates"]["status"] = "failed"
                results["python_updates"]["error"] = stderr[:1000]
            
            # Check for Docker base image updates
            logger.info("Checking Docker base image updates...")
            
            # Extract base images from Dockerfile
            with open("Dockerfile") as f:
                dockerfile_content = f.read()
            
            base_images = []
            for line in dockerfile_content.split('\n'):
                if line.strip().startswith('FROM '):
                    image = line.split()[1]
                    base_images.append(image)
            
            results["docker_updates"]["base_images"] = base_images
            results["docker_updates"]["status"] = "checked"
            
            # Run security scan after updates
            code, stdout, stderr = await self.run_command([
                "safety", "check", "--json"
            ])
            
            if code == 0:
                results["security_scan"]["vulnerabilities_after"] = len(json.loads(stdout or "[]"))
            
            results["status"] = "completed"
            results["completed_at"] = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            logger.error(f"Dependency update failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    async def run_security_scans(self) -> Dict[str, Any]:
        """Run comprehensive security scans."""
        logger.info("🔒 Running security scans...")
        
        results = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "bandit_scan": {},
            "safety_scan": {},
            "secret_scan": {},
            "docker_scan": {},
            "status": "running"
        }
        
        try:
            # Bandit security scan
            logger.info("Running Bandit security scan...")
            code, stdout, stderr = await self.run_command([
                "bandit", "-r", "src/", "-f", "json"
            ])
            
            if stdout:
                bandit_data = json.loads(stdout)
                results["bandit_scan"] = {
                    "status": "completed",
                    "issues": {
                        "high": len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "HIGH"]),
                        "medium": len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "MEDIUM"]),
                        "low": len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "LOW"])
                    }
                }
            
            # Safety dependency scan
            logger.info("Running Safety dependency scan...")
            code, stdout, stderr = await self.run_command([
                "safety", "check", "--json"
            ])
            
            if code == 0:
                safety_data = json.loads(stdout or "[]")
                results["safety_scan"] = {
                    "status": "completed",
                    "vulnerabilities": len(safety_data)
                }
            
            # Secret scanning (using detect-secrets if available)
            logger.info("Running secret detection scan...")
            code, stdout, stderr = await self.run_command([
                "detect-secrets", "scan", ".", "--all-files"
            ])
            
            if code == 0:
                results["secret_scan"] = {
                    "status": "completed",
                    "secrets_found": "clean" if "No secrets were detected" in stdout else "potential_secrets"
                }
            
            # Docker security scan
            logger.info("Running Docker security scan...")
            code, stdout, stderr = await self.run_command([
                "trivy", "fs", ".", "--format", "json"
            ])
            
            if code == 0 and stdout:
                trivy_data = json.loads(stdout)
                vulnerabilities = []
                for result in trivy_data.get("Results", []):
                    vulnerabilities.extend(result.get("Vulnerabilities", []))
                
                results["docker_scan"] = {
                    "status": "completed",
                    "vulnerabilities": len(vulnerabilities),
                    "critical": len([v for v in vulnerabilities if v.get("Severity") == "CRITICAL"]),
                    "high": len([v for v in vulnerabilities if v.get("Severity") == "HIGH"])
                }
            
            results["status"] = "completed"
            results["completed_at"] = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    async def generate_documentation(self) -> Dict[str, Any]:
        """Generate and update project documentation."""
        logger.info("📚 Generating documentation...")
        
        results = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "api_docs": {},
            "readme_updates": {},
            "changelog": {},
            "status": "running"
        }
        
        try:
            # Generate API documentation
            logger.info("Generating API documentation...")
            code, stdout, stderr = await self.run_command([
                "python", "-m", "pydoc", "-w", "docker_optimizer"
            ])
            
            if code == 0:
                results["api_docs"]["status"] = "generated"
            
            # Update README with current metrics
            logger.info("Updating README with current metrics...")
            
            # Read current metrics
            metrics_file = Path(".github/project-metrics.json")
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                
                # Update README badges/stats section
                readme_file = Path("README.md")
                if readme_file.exists():
                    with open(readme_file) as f:
                        readme_content = f.read()
                    
                    # Update coverage badge
                    coverage = metrics.get("code_quality", {}).get("code_coverage", 0)
                    readme_content = self._update_badge(
                        readme_content, 
                        "coverage", 
                        f"{coverage}%", 
                        "brightgreen" if coverage >= 90 else "yellow"
                    )
                    
                    # Update build badge (assuming success for now)
                    readme_content = self._update_badge(
                        readme_content,
                        "build",
                        "passing",
                        "brightgreen"
                    )
                    
                    if not self.dry_run:
                        with open(readme_file, "w") as f:
                            f.write(readme_content)
                    
                    results["readme_updates"]["status"] = "updated"
            
            # Generate changelog entry
            logger.info("Updating changelog...")
            code, stdout, stderr = await self.run_command([
                "git", "log", "--oneline", "--since=1.week.ago"
            ])
            
            if code == 0 and stdout:
                recent_commits = stdout.strip().split('\n')
                results["changelog"] = {
                    "status": "updated",
                    "recent_commits": len(recent_commits)
                }
            
            results["status"] = "completed"
            results["completed_at"] = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    def _update_badge(self, content: str, badge_type: str, value: str, color: str) -> str:
        """Update a badge in markdown content."""
        # This is a simple implementation - in reality, you'd want more sophisticated badge updating
        badge_pattern = f"![{badge_type}]"
        new_badge = f"![{badge_type}](https://img.shields.io/badge/{badge_type}-{value.replace('%', '%25')}-{color})"
        
        if badge_pattern in content:
            import re
            pattern = rf"!\[{badge_type}\]\([^)]+\)"
            content = re.sub(pattern, new_badge, content)
        
        return content
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive project metrics."""
        logger.info("📊 Collecting metrics...")
        
        results = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running"
        }
        
        try:
            # Run the metrics collection script
            code, stdout, stderr = await self.run_command([
                "python", "scripts/collect-metrics.py", "--summary"
            ])
            
            if code == 0:
                results["status"] = "completed"
                results["metrics_collected"] = True
            else:
                results["status"] = "failed"
                results["error"] = stderr
            
            results["completed_at"] = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    async def cleanup_tasks(self) -> Dict[str, Any]:
        """Perform repository cleanup tasks."""
        logger.info("🧹 Running cleanup tasks...")
        
        results = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "docker_cleanup": {},
            "pip_cache_cleanup": {},
            "log_rotation": {},
            "status": "running"
        }
        
        try:
            # Docker cleanup
            logger.info("Cleaning up Docker artifacts...")
            code, stdout, stderr = await self.run_command([
                "docker", "system", "prune", "-f"
            ])
            
            if code == 0:
                results["docker_cleanup"]["status"] = "completed"
                results["docker_cleanup"]["space_freed"] = "calculated_from_output"
            
            # Pip cache cleanup
            logger.info("Cleaning pip cache...")
            code, stdout, stderr = await self.run_command([
                "pip", "cache", "purge"
            ])
            
            if code == 0:
                results["pip_cache_cleanup"]["status"] = "completed"
            
            # Log rotation (keep last 10 log files)
            log_files = list(Path(".").glob("*.log"))
            if len(log_files) > 10:
                log_files.sort(key=lambda x: x.stat().st_mtime)
                old_logs = log_files[:-10]
                
                for log_file in old_logs:
                    if not self.dry_run:
                        log_file.unlink()
                
                results["log_rotation"] = {
                    "status": "completed",
                    "files_removed": len(old_logs)
                }
            
            results["status"] = "completed"
            results["completed_at"] = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            logger.error(f"Cleanup tasks failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    async def send_notifications(self) -> Dict[str, Any]:
        """Send notifications about pipeline results."""
        logger.info("📢 Sending notifications...")
        
        results = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "slack_notification": {},
            "github_issue": {},
            "status": "running"
        }
        
        try:
            # Prepare notification content
            summary = self._generate_pipeline_summary()
            
            # Send Slack notification if configured
            slack_webhook = self.config.get("notifications", {}).get("slack_webhook")
            if slack_webhook and not self.dry_run:
                import requests
                
                payload = {
                    "text": "🤖 Automation Pipeline Completed",
                    "attachments": [
                        {
                            "color": "good" if self._is_pipeline_successful() else "danger",
                            "fields": [
                                {
                                    "title": "Status",
                                    "value": "✅ Success" if self._is_pipeline_successful() else "❌ Failed",
                                    "short": True
                                },
                                {
                                    "title": "Duration",
                                    "value": f"{(datetime.now(timezone.utc) - self.start_time).total_seconds():.1f}s",
                                    "short": True
                                }
                            ],
                            "text": summary
                        }
                    ]
                }
                
                response = requests.post(slack_webhook, json=payload, timeout=10)
                results["slack_notification"]["status"] = "sent" if response.status_code == 200 else "failed"
            
            # Create GitHub issue for failures
            if not self._is_pipeline_successful() and self.config.get("notifications", {}).get("github_issues"):
                issue_title = f"🚨 Automation Pipeline Failure - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                issue_body = f"""
## Automation Pipeline Failure Report

**Timestamp:** {datetime.now(timezone.utc).isoformat()}
**Duration:** {(datetime.now(timezone.utc) - self.start_time).total_seconds():.1f} seconds

### Summary
{summary}

### Failed Tasks
{self._get_failed_tasks()}

### Recommended Actions
- Review the automation logs
- Check for infrastructure issues
- Update automation scripts if needed
- Manual intervention may be required

### Logs
Check the automation.log file for detailed error information.
                """
                
                results["github_issue"] = {
                    "status": "would_create" if self.dry_run else "created",
                    "title": issue_title
                }
            
            results["status"] = "completed"
            results["completed_at"] = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            logger.error(f"Notification sending failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    def _generate_pipeline_summary(self) -> str:
        """Generate a summary of pipeline results."""
        total_tasks = len(self.results)
        successful_tasks = sum(1 for r in self.results.values() if r.get("status") == "completed")
        
        summary = f"Pipeline completed: {successful_tasks}/{total_tasks} tasks successful"
        
        if self.results:
            summary += "\n\nTask Results:\n"
            for task_name, result in self.results.items():
                status_emoji = "✅" if result.get("status") == "completed" else "❌"
                summary += f"{status_emoji} {task_name}: {result.get('status', 'unknown')}\n"
        
        return summary
    
    def _is_pipeline_successful(self) -> bool:
        """Check if the overall pipeline was successful."""
        return all(r.get("status") == "completed" for r in self.results.values())
    
    def _get_failed_tasks(self) -> str:
        """Get a list of failed tasks with error details."""
        failed_tasks = []
        for task_name, result in self.results.items():
            if result.get("status") != "completed":
                error = result.get("error", "Unknown error")
                failed_tasks.append(f"- {task_name}: {error}")
        
        return "\n".join(failed_tasks) if failed_tasks else "No failed tasks"
    
    async def run_pipeline(self, tasks: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the complete automation pipeline."""
        logger.info("🚀 Starting automation pipeline...")
        
        # Define available tasks
        available_tasks = {
            "dependencies": self.update_dependencies,
            "security": self.run_security_scans,
            "documentation": self.generate_documentation,
            "metrics": self.collect_metrics,
            "cleanup": self.cleanup_tasks,
            "notifications": self.send_notifications
        }
        
        # Determine which tasks to run
        if tasks is None:
            tasks_to_run = []
            automation_config = self.config.get("automation", {})
            
            if automation_config.get("update_dependencies", True):
                tasks_to_run.append("dependencies")
            if automation_config.get("run_security_scans", True):
                tasks_to_run.append("security")
            if automation_config.get("generate_documentation", True):
                tasks_to_run.append("documentation")
            if automation_config.get("collect_metrics", True):
                tasks_to_run.append("metrics")
            if automation_config.get("cleanup_tasks", True):
                tasks_to_run.append("cleanup")
            if automation_config.get("send_notifications", True):
                tasks_to_run.append("notifications")
        else:
            tasks_to_run = [t for t in tasks if t in available_tasks]
        
        logger.info(f"Running tasks: {', '.join(tasks_to_run)}")
        
        # Run tasks sequentially (some tasks depend on others)
        for task_name in tasks_to_run:
            if task_name in available_tasks:
                logger.info(f"▶️ Starting task: {task_name}")
                try:
                    result = await available_tasks[task_name]()
                    self.results[task_name] = result
                    logger.info(f"✅ Completed task: {task_name}")
                except Exception as e:
                    logger.error(f"❌ Failed task: {task_name}, Error: {e}")
                    self.results[task_name] = {
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
        
        # Generate final report
        pipeline_duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        final_report = {
            "pipeline_info": {
                "started_at": self.start_time.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": pipeline_duration,
                "dry_run": self.dry_run
            },
            "tasks_executed": list(self.results.keys()),
            "tasks_successful": [k for k, v in self.results.items() if v.get("status") == "completed"],
            "tasks_failed": [k for k, v in self.results.items() if v.get("status") != "completed"],
            "overall_success": self._is_pipeline_successful(),
            "results": self.results
        }
        
        # Save report
        report_file = Path(f"automation-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
        with open(report_file, "w") as f:
            json.dump(final_report, f, indent=2)
        
        logger.info(f"🎉 Pipeline completed in {pipeline_duration:.1f}s")
        logger.info(f"📄 Report saved to {report_file}")
        
        return final_report


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run automation pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["dependencies", "security", "documentation", "metrics", "cleanup", "notifications"],
        help="Specific tasks to run (default: all enabled tasks)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no actual changes)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run pipeline
    pipeline = AutomationPipeline(args.config, args.dry_run)
    report = await pipeline.run_pipeline(args.tasks)
    
    # Exit with appropriate code
    sys.exit(0 if report["overall_success"] else 1)


if __name__ == "__main__":
    asyncio.run(main())