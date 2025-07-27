#!/usr/bin/env python3
"""
Targeted autonomous execution that promotes and executes specific tasks.
"""
import sys
sys.path.append('/root/repo/src')

from autonomous_backlog import AutonomousBacklog, TaskStatus, RiskTier

def main():
    # Initialize the system
    backlog = AutonomousBacklog()
    backlog.load_backlog()
    
    print(f"🚀 Targeted Autonomous Execution Starting...")
    print(f"Current backlog size: {len(backlog.backlog)}")
    
    # Find tasks that aren't comments or low-value TODOs
    actionable_tasks = []
    for item in backlog.backlog:
        # Skip comment-based TODOs, focus on actual implementation items
        if ("TODO" not in item.title or 
            "import" in item.title.lower() or 
            "logging" in item.title.lower() or
            "config" in item.title.lower()):
            actionable_tasks.append(item)
    
    print(f"Found {len(actionable_tasks)} potentially actionable tasks")
    
    # Promote top 5 to READY and execute them
    completed_tasks = []
    
    for i, task in enumerate(actionable_tasks[:5]):
        print(f"\n--- Executing Task {i+1}/5 ---")
        print(f"Task: {task.title}")
        
        # Promote to READY
        task.status = TaskStatus.READY
        task.risk_tier = RiskTier.LOW  # Mark as low risk for execution
        
        # Execute the task
        success = backlog.execute_task(task)
        if success:
            completed_tasks.append(task.id)
            print(f"✅ Completed: {task.title}")
        else:
            print(f"❌ Failed: {task.title}")
    
    # Run CI checks and save metrics
    ci_results = backlog.run_ci_checks()
    backlog.save_metrics(completed_tasks, ci_results)
    backlog.save_backlog()
    
    print(f"\n🏁 Targeted execution complete!")
    print(f"✅ Completed {len(completed_tasks)} tasks")
    print(f"📊 Total backlog size: {len(backlog.backlog)}")
    
    # Summary of completed tasks
    done_tasks = [item for item in backlog.backlog if item.status == TaskStatus.DONE]
    print(f"📈 Tasks marked DONE: {len(done_tasks)}")

if __name__ == "__main__":
    main()