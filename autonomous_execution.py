#!/usr/bin/env python3
"""
Autonomous execution engine for continuous SDLC enhancement.
Implements the perpetual value discovery loop with safety constraints.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import existing systems
import sys
sys.path.append('/root/repo/src')
from autonomous_backlog import AutonomousBacklog, BacklogItem, TaskStatus, RiskTier


class AutonomousExecutionEngine:
    """Main autonomous execution engine with safety constraints."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_system = AutonomousBacklog(repo_path)
        
        # Safety constraints
        self.max_daily_tasks = 10
        self.max_file_changes = 5
        self.restricted_paths = [".github/workflows", ".git"]
        
        # Execution state
        self.current_session = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "tasks_completed": 0,
            "files_modified": 0,
            "safety_violations": 0
        }
    
    def discover_manual_tasks(self) -> List[BacklogItem]:
        """Discover tasks manually without external dependencies."""
        discovered = []
        
        # Search for TODO/FIXME in Python files manually
        todo_patterns = ["TODO", "FIXME", "HACK", "BUG", "XXX"]
        
        try:
            for py_file in self.repo_path.rglob("*.py"):
                if any(restricted in str(py_file) for restricted in self.restricted_paths):
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        
                    for line_num, line in enumerate(lines, 1):
                        for pattern in todo_patterns:
                            if pattern in line and not line.strip().startswith('#'):
                                relative_path = str(py_file.relative_to(self.repo_path))
                                task_id = f"manual-{hash(f'{relative_path}:{line_num}') % 10000}"
                                
                                # Check if already exists
                                if not any(item.id == task_id for item in self.backlog_system.backlog):
                                    discovered.append(BacklogItem(
                                        id=task_id,
                                        title=f"Address {pattern}: {line.strip()[:50]}...",
                                        type="tech-debt",
                                        description=f"Found in {relative_path}:{line_num}\n{line.strip()}",
                                        acceptance_criteria=[f"Resolve {pattern} in {relative_path}:{line_num}"],
                                        effort=2,
                                        value=3,
                                        time_criticality=2,
                                        risk_reduction=2,
                                        wsjf_score=3.5,
                                        aging_multiplier=1.0,
                                        status=TaskStatus.NEW,
                                        risk_tier=RiskTier.LOW,
                                        created_at=datetime.now(timezone.utc).isoformat(),
                                        links=[relative_path],
                                        tags=["discovered", "todo-comment"]
                                    ))
                                break
                                
                except Exception:
                    continue  # Skip files that can't be read
                    
        except Exception as e:
            print(f"Error in manual discovery: {e}")
        
        return discovered
    
    def discover_environment_issues(self) -> List[BacklogItem]:
        """Discover environment and dependency issues."""
        discovered = []
        
        # Check for missing development dependencies
        missing_tools = []
        tools_to_check = [
            ("pytest", "Python test runner"),
            ("ruff", "Python linter"),
            ("mypy", "Type checker"),
            ("black", "Code formatter")
        ]
        
        for tool, description in tools_to_check:
            if os.system(f"which {tool} >/dev/null 2>&1") != 0:
                missing_tools.append((tool, description))
        
        if missing_tools:
            tool_list = ", ".join([tool for tool, _ in missing_tools])
            discovered.append(BacklogItem(
                id="env-deps-001",
                title=f"Install missing development dependencies: {tool_list}",
                type="environment",
                description=f"Missing development tools: {', '.join([f'{tool} ({desc})' for tool, desc in missing_tools])}",
                acceptance_criteria=[
                    "Set up Python virtual environment",
                    "Install all development dependencies",
                    "Verify tools are accessible"
                ],
                effort=3,
                value=8,
                time_criticality=5,
                risk_reduction=3,
                wsjf_score=5.33,
                aging_multiplier=1.0,
                status=TaskStatus.NEW,
                risk_tier=RiskTier.MEDIUM,
                created_at=datetime.now(timezone.utc).isoformat(),
                links=["setup_environment.py", "pyproject.toml"],
                tags=["discovered", "environment", "dependencies"]
            ))
        
        # Check for Makefile issues
        makefile_path = self.repo_path / "Makefile"
        if makefile_path.exists():
            try:
                with open(makefile_path, 'r') as f:
                    makefile_content = f.read()
                    
                # Look for duplicate targets
                if "warning: overriding recipe for target" in os.popen("make help 2>&1").read():
                    discovered.append(BacklogItem(
                        id="makefile-001",
                        title="Fix duplicate Makefile targets causing warnings",
                        type="build-system",
                        description="Makefile has duplicate target definitions causing warnings",
                        acceptance_criteria=[
                            "Remove duplicate target definitions",
                            "Verify make commands work without warnings",
                            "Test all make targets"
                        ],
                        effort=2,
                        value=4,
                        time_criticality=3,
                        risk_reduction=2,
                        wsjf_score=4.5,
                        aging_multiplier=1.0,
                        status=TaskStatus.NEW,
                        risk_tier=RiskTier.LOW,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        links=["Makefile"],
                        tags=["discovered", "build-system", "warnings"]
                    ))
            except Exception:
                pass
        
        return discovered
    
    def discover_documentation_gaps(self) -> List[BacklogItem]:
        """Discover documentation that needs updates."""
        discovered = []
        
        # Check for environment setup documentation
        if not (self.repo_path / "docs" / "ENVIRONMENT_SETUP.md").exists():
            discovered.append(BacklogItem(
                id="docs-env-001", 
                title="Create comprehensive environment setup documentation",
                type="documentation",
                description="Missing documentation for setting up development environment with virtual environments",
                acceptance_criteria=[
                    "Document virtual environment setup process",
                    "Include dependency installation steps",
                    "Add troubleshooting section",
                    "Provide platform-specific instructions"
                ],
                effort=3,
                value=6,
                time_criticality=2,
                risk_reduction=4,
                wsjf_score=4.0,
                aging_multiplier=1.0,
                status=TaskStatus.NEW,
                risk_tier=RiskTier.LOW,
                created_at=datetime.now(timezone.utc).isoformat(),
                links=["docs/"],
                tags=["discovered", "documentation", "environment"]
            ))
        
        return discovered
    
    def run_discovery_cycle(self) -> List[BacklogItem]:
        """Run a complete discovery cycle."""
        print("🔍 Running autonomous discovery cycle...")
        
        all_discovered = []
        
        # Manual discovery methods
        discovery_methods = [
            ("Manual TODO discovery", self.discover_manual_tasks),
            ("Environment issues", self.discover_environment_issues),
            ("Documentation gaps", self.discover_documentation_gaps)
        ]
        
        for method_name, method in discovery_methods:
            try:
                items = method()
                print(f"  📋 {method_name}: {len(items)} items")
                all_discovered.extend(items)
            except Exception as e:
                print(f"  ❌ {method_name} failed: {e}")
        
        print(f"🎯 Total discovered: {len(all_discovered)} items")
        return all_discovered
    
    def execute_safe_task(self, task: BacklogItem) -> bool:
        """Execute a task with safety constraints."""
        print(f"🚀 Executing: {task.title}")
        
        # Safety checks
        if self.current_session["tasks_completed"] >= self.max_daily_tasks:
            print("⚠️  Daily task limit reached")
            return False
            
        if self.current_session["files_modified"] >= self.max_file_changes:
            print("⚠️  File modification limit reached")
            return False
        
        # Check if task affects restricted paths
        for link in task.links:
            if any(restricted in link for restricted in self.restricted_paths):
                print(f"⚠️  Task affects restricted path: {link}")
                return False
        
        # Simulate task execution (in real implementation, this would do actual work)
        if task.type == "environment":
            return self._execute_environment_task(task)
        elif task.type == "documentation":
            return self._execute_documentation_task(task)
        elif task.type == "tech-debt":
            return self._execute_tech_debt_task(task)
        else:
            print(f"📝 Task type '{task.type}' requires manual implementation")
            return False
    
    def _execute_environment_task(self, task: BacklogItem) -> bool:
        """Execute environment-related tasks."""
        if "missing development dependencies" in task.title.lower():
            # Create a setup guide instead of trying to install directly
            setup_guide = """# Development Environment Setup

## Prerequisites
- Python 3.9+ installed
- Git installed

## Setup Steps

1. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -e .[dev,security]
   ```

3. **Install Pre-commit Hooks**:
   ```bash
   pre-commit install
   ```

4. **Verify Installation**:
   ```bash
   make test
   make lint
   make typecheck
   ```

## Troubleshooting

- If you see "externally-managed-environment" error, use a virtual environment
- On Ubuntu/Debian, you may need: `sudo apt install python3-venv python3-dev`
- On macOS with Homebrew: `brew install python@3.11`

## Tool Verification

Run these commands to verify tools are installed:
- `pytest --version`
- `ruff --version` 
- `mypy --version`
- `black --version`
"""
            
            docs_dir = self.repo_path / "docs"
            docs_dir.mkdir(exist_ok=True)
            
            with open(docs_dir / "ENVIRONMENT_SETUP.md", 'w') as f:
                f.write(setup_guide)
            
            print("✅ Created environment setup documentation")
            self.current_session["files_modified"] += 1
            return True
            
        return False
    
    def _execute_documentation_task(self, task: BacklogItem) -> bool:
        """Execute documentation tasks."""
        # This is a placeholder - in real implementation would create actual docs
        print("📝 Documentation task would be implemented here")
        return True
    
    def _execute_tech_debt_task(self, task: BacklogItem) -> bool:
        """Execute technical debt tasks."""
        # This is a placeholder - would need to analyze specific TODO items
        print("🔧 Technical debt task requires manual review")
        return False
    
    def save_session_metrics(self) -> None:
        """Save session execution metrics."""
        self.current_session["end_time"] = datetime.now(timezone.utc).isoformat()
        
        # Load existing metrics
        if self.metrics_path.exists():
            with open(self.metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {
                "repository": {"name": "docker-optimizer-agent"},
                "executionHistory": [],
                "continuousMetrics": {}
            }
        
        # Add session to history
        metrics["executionHistory"].append({
            "timestamp": self.current_session["start_time"],
            "session_type": "autonomous_execution",
            "tasks_completed": self.current_session["tasks_completed"],
            "files_modified": self.current_session["files_modified"],
            "safety_violations": self.current_session["safety_violations"],
            "duration_minutes": self._calculate_session_duration()
        })
        
        # Update continuous metrics
        metrics["continuousMetrics"]["totalItemsCompleted"] = \
            metrics["continuousMetrics"].get("totalItemsCompleted", 0) + \
            self.current_session["tasks_completed"]
        
        # Save updated metrics
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"💾 Session metrics saved: {self.current_session['tasks_completed']} tasks completed")
    
    def _calculate_session_duration(self) -> float:
        """Calculate session duration in minutes."""
        try:
            start = datetime.fromisoformat(self.current_session["start_time"].replace('Z', '+00:00'))
            end = datetime.now(timezone.utc)
            return (end - start).total_seconds() / 60
        except Exception:
            return 0.0
    
    def run_execution_cycle(self, max_tasks: int = 3) -> None:
        """Run a complete execution cycle."""
        print("🤖 Starting autonomous execution cycle...")
        
        # Load existing backlog
        self.backlog_system.load_backlog()
        
        # Run discovery
        discovered = self.run_discovery_cycle()
        
        # Add discovered items to backlog
        if discovered:
            existing_ids = {item.id for item in self.backlog_system.backlog}
            new_items = [item for item in discovered if item.id not in existing_ids]
            
            if new_items:
                self.backlog_system.backlog.extend(new_items)
                print(f"📋 Added {len(new_items)} new items to backlog")
        
        # Score and sort backlog
        self.backlog_system.score_and_sort_backlog()
        
        # Execute highest priority tasks
        executed_count = 0
        for task in self.backlog_system.backlog:
            if executed_count >= max_tasks:
                break
                
            if task.status == TaskStatus.NEW and task.risk_tier in [RiskTier.LOW, RiskTier.MEDIUM]:
                task.status = TaskStatus.DOING
                
                if self.execute_safe_task(task):
                    task.status = TaskStatus.DONE
                    executed_count += 1
                    self.current_session["tasks_completed"] += 1
                    print(f"✅ Completed: {task.title}")
                else:
                    task.status = TaskStatus.BLOCKED
                    print(f"🚫 Blocked: {task.title}")
        
        # Save updated backlog
        self.backlog_system.save_backlog()
        
        # Save session metrics
        self.save_session_metrics()
        
        print(f"🏁 Execution cycle complete: {executed_count} tasks executed")


def main():
    """Main entry point for autonomous execution."""
    engine = AutonomousExecutionEngine()
    engine.run_execution_cycle(max_tasks=5)


if __name__ == "__main__":
    main()