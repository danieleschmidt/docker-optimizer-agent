#!/usr/bin/env python3
"""
Autonomous scheduler for continuous SDLC execution.
Implements multiple execution schedules based on triggers and intervals.
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Import execution components
from autonomous_execution import AutonomousExecutionEngine


class AutonomousScheduler:
    """Scheduler for autonomous SDLC execution cycles."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.state_path = self.repo_path / ".terragon" / "scheduler-state.json"
        self.execution_engine = AutonomousExecutionEngine(repo_path)
        
        # Schedule configuration
        self.schedules = {
            "immediate": {"interval": 0, "last_run": None, "enabled": True},
            "hourly": {"interval": 3600, "last_run": None, "enabled": True},
            "daily": {"interval": 86400, "last_run": None, "enabled": True},
            "weekly": {"interval": 604800, "last_run": None, "enabled": True},
            "monthly": {"interval": 2592000, "last_run": None, "enabled": False}
        }
        
        self.load_scheduler_state()
    
    def load_scheduler_state(self) -> None:
        """Load scheduler state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    state = json.load(f)
                    
                for schedule_name in self.schedules:
                    if schedule_name in state:
                        self.schedules[schedule_name].update(state[schedule_name])
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}")
    
    def save_scheduler_state(self) -> None:
        """Save scheduler state to disk."""
        try:
            with open(self.state_path, 'w') as f:
                json.dump(self.schedules, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save scheduler state: {e}")
    
    def should_run_schedule(self, schedule_name: str) -> bool:
        """Check if a schedule should run based on interval."""
        schedule = self.schedules[schedule_name]
        
        if not schedule["enabled"]:
            return False
            
        if schedule["last_run"] is None:
            return True
            
        try:
            last_run = datetime.fromisoformat(schedule["last_run"].replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            elapsed = (now - last_run).total_seconds()
            
            return elapsed >= schedule["interval"]
        except Exception:
            return True
    
    def execute_immediate_cycle(self) -> Dict:
        """Execute immediate discovery and high-priority tasks."""
        print("🚀 Running immediate execution cycle...")
        
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "immediate",
            "tasks_completed": 0,
            "items_discovered": 0,
            "status": "completed"
        }
        
        try:
            # Run quick discovery and execution
            self.execution_engine.current_session = {
                "start_time": datetime.now(timezone.utc).isoformat(),
                "tasks_completed": 0,
                "files_modified": 0,
                "safety_violations": 0
            }
            
            # Load backlog and run discovery
            self.execution_engine.backlog_system.load_backlog()
            discovered = self.execution_engine.run_discovery_cycle()
            results["items_discovered"] = len(discovered)
            
            # Execute up to 2 high-priority tasks for immediate cycle
            if discovered:
                existing_ids = {item.id for item in self.execution_engine.backlog_system.backlog}
                new_items = [item for item in discovered if item.id not in existing_ids]
                
                if new_items:
                    self.execution_engine.backlog_system.backlog.extend(new_items[:5])  # Limit immediate additions
                    self.execution_engine.backlog_system.score_and_sort_backlog()
                    self.execution_engine.backlog_system.save_backlog()
            
            # Execute top 2 tasks
            executed = 0
            for task in self.execution_engine.backlog_system.backlog[:2]:
                if task.status.value == "NEW" and executed < 2:
                    task.status = self.execution_engine.backlog_system.TaskStatus.DOING
                    
                    if self.execution_engine.execute_safe_task(task):
                        task.status = self.execution_engine.backlog_system.TaskStatus.DONE
                        executed += 1
                        self.execution_engine.current_session["tasks_completed"] += 1
                    else:
                        task.status = self.execution_engine.backlog_system.TaskStatus.BLOCKED
            
            results["tasks_completed"] = executed
            self.execution_engine.save_session_metrics()
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            print(f"❌ Immediate cycle failed: {e}")
        
        return results
    
    def execute_hourly_cycle(self) -> Dict:
        """Execute hourly security and dependency scans."""
        print("🔍 Running hourly security cycle...")
        
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "hourly",
            "security_issues": 0,
            "dependency_updates": 0,
            "status": "completed"
        }
        
        try:
            # Focus on security and dependency discovery
            discovered = []
            
            # Check for security patterns (simplified)
            security_items = self.execution_engine.discover_environment_issues()
            discovered.extend(security_items)
            results["security_issues"] = len(security_items)
            
            # Add to backlog if new items found
            if discovered:
                self.execution_engine.backlog_system.load_backlog()
                existing_ids = {item.id for item in self.execution_engine.backlog_system.backlog}
                new_items = [item for item in discovered if item.id not in existing_ids]
                
                if new_items:
                    self.execution_engine.backlog_system.backlog.extend(new_items)
                    self.execution_engine.backlog_system.score_and_sort_backlog()
                    self.execution_engine.backlog_system.save_backlog()
                    
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            print(f"❌ Hourly cycle failed: {e}")
        
        return results
    
    def execute_daily_cycle(self) -> Dict:
        """Execute comprehensive daily analysis."""
        print("📊 Running daily comprehensive cycle...")
        
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "daily",
            "analysis_completed": False,
            "backlog_size": 0,
            "status": "completed"
        }
        
        try:
            # Run full discovery and analysis
            self.execution_engine.run_execution_cycle(max_tasks=5)
            
            # Get backlog metrics
            self.execution_engine.backlog_system.load_backlog()
            results["backlog_size"] = len(self.execution_engine.backlog_system.backlog)
            results["analysis_completed"] = True
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            print(f"❌ Daily cycle failed: {e}")
        
        return results
    
    def execute_weekly_cycle(self) -> Dict:
        """Execute weekly deep analysis and optimization."""
        print("🏗️ Running weekly optimization cycle...")
        
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "weekly",
            "optimizations_applied": 0,
            "architecture_review": False,
            "status": "completed"
        }
        
        try:
            # Deep analysis and optimization
            self.execution_engine.run_execution_cycle(max_tasks=10)
            results["optimizations_applied"] = self.execution_engine.current_session["tasks_completed"]
            results["architecture_review"] = True
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            print(f"❌ Weekly cycle failed: {e}")
        
        return results
    
    def run_scheduled_cycles(self) -> List[Dict]:
        """Run all scheduled cycles that are due."""
        results = []
        
        # Check each schedule
        schedule_runners = {
            "immediate": self.execute_immediate_cycle,
            "hourly": self.execute_hourly_cycle,
            "daily": self.execute_daily_cycle,
            "weekly": self.execute_weekly_cycle
        }
        
        for schedule_name, runner in schedule_runners.items():
            if self.should_run_schedule(schedule_name):
                print(f"⏰ Executing {schedule_name} cycle...")
                
                try:
                    result = runner()
                    results.append(result)
                    
                    # Update last run time
                    self.schedules[schedule_name]["last_run"] = datetime.now(timezone.utc).isoformat()
                    
                except Exception as e:
                    error_result = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "type": schedule_name,
                        "status": "failed",
                        "error": str(e)
                    }
                    results.append(error_result)
                    print(f"❌ {schedule_name} cycle failed: {e}")
        
        # Save scheduler state
        self.save_scheduler_state()
        
        return results
    
    def run_continuous_mode(self, duration_hours: int = 24) -> None:
        """Run continuous scheduling for specified duration."""
        print(f"🔄 Starting continuous mode for {duration_hours} hours...")
        
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(hours=duration_hours)
        
        execution_count = 0
        
        while datetime.now(timezone.utc) < end_time:
            # Check for scheduled cycles
            results = self.run_scheduled_cycles()
            
            if results:
                execution_count += len(results)
                print(f"📈 Executed {len(results)} cycles (total: {execution_count})")
                
                # Log results
                for result in results:
                    if result["status"] == "completed":
                        print(f"  ✅ {result['type']} cycle completed")
                    else:
                        print(f"  ❌ {result['type']} cycle failed")
            
            # Sleep for 1 minute between checks
            time.sleep(60)
        
        print(f"🏁 Continuous mode completed: {execution_count} total executions")
    
    def get_schedule_status(self) -> Dict:
        """Get current schedule status."""
        status = {
            "current_time": datetime.now(timezone.utc).isoformat(),
            "schedules": {}
        }
        
        for name, schedule in self.schedules.items():
            next_run = "now" if self.should_run_schedule(name) else "scheduled"
            if schedule["last_run"]:
                try:
                    last_run = datetime.fromisoformat(schedule["last_run"].replace('Z', '+00:00'))
                    next_run_time = last_run + timedelta(seconds=schedule["interval"])
                    next_run = next_run_time.isoformat()
                except Exception:
                    next_run = "unknown"
            
            status["schedules"][name] = {
                "enabled": schedule["enabled"],
                "interval_hours": schedule["interval"] / 3600,
                "last_run": schedule["last_run"],
                "next_run": next_run,
                "due": self.should_run_schedule(name)
            }
        
        return status


def main():
    """Main entry point for autonomous scheduler."""
    import sys
    
    scheduler = AutonomousScheduler()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "status":
            status = scheduler.get_schedule_status()
            print(json.dumps(status, indent=2))
            
        elif mode == "continuous":
            hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
            scheduler.run_continuous_mode(hours)
            
        elif mode == "once":
            results = scheduler.run_scheduled_cycles()
            print(f"Executed {len(results)} cycles")
            
        else:
            print("Usage: python3 autonomous_scheduler.py [status|continuous [hours]|once]")
    else:
        # Default: run once
        results = scheduler.run_scheduled_cycles()
        print(f"✅ Executed {len(results)} scheduled cycles")


if __name__ == "__main__":
    main()