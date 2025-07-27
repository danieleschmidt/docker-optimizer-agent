#!/usr/bin/env python3
"""
Run discovery and promote tasks to actionable status.
"""
import sys
sys.path.append('/root/repo/src')

from autonomous_backlog import AutonomousBacklog, TaskStatus

def main():
    # Load the autonomous backlog system
    backlog = AutonomousBacklog()
    backlog.load_backlog()
    
    print(f"Current backlog has {len(backlog.backlog)} items")
    
    # Discover new tasks
    discovered = backlog.discover_tasks()
    print(f"Discovered {len(discovered)} new tasks")
    
    # Add discovered tasks to backlog
    backlog.backlog.extend(discovered)
    
    # Promote first 3 discovered tasks to READY status
    promoted_count = 0
    for task in discovered[:3]:
        task.status = TaskStatus.READY
        promoted_count += 1
        print(f"Promoted to READY: {task.title}")
    
    # Score and sort
    backlog.score_and_sort_backlog()
    
    # Save the updated backlog
    backlog.save_backlog()
    
    print(f"Total backlog size: {len(backlog.backlog)}")
    print(f"Promoted {promoted_count} tasks to READY status")
    
    # Show READY tasks
    ready_tasks = [item for item in backlog.backlog if item.status == TaskStatus.READY]
    print(f"READY tasks: {len(ready_tasks)}")
    for task in ready_tasks:
        print(f"  - {task.title} (WSJF: {task.wsjf_score:.2f})")

if __name__ == "__main__":
    main()