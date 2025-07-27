#!/usr/bin/env python3
"""
Simple script to promote discovered TODO items to READY status for execution.
"""
import sys
sys.path.append('/root/repo/src')

from autonomous_backlog import AutonomousBacklog, TaskStatus

def main():
    # Load the autonomous backlog system
    backlog = AutonomousBacklog()
    backlog.load_backlog()
    
    # Find NEW tasks and promote a few high-value ones to READY
    new_tasks = [item for item in backlog.backlog if item.status == TaskStatus.NEW]
    
    print(f"Found {len(new_tasks)} NEW tasks")
    
    # Promote up to 3 tasks to READY status
    promoted_count = 0
    for task in new_tasks[:3]:
        if promoted_count < 3:
            task.status = TaskStatus.READY
            promoted_count += 1
            print(f"Promoted task to READY: {task.title}")
    
    # Save the updated backlog
    backlog.save_backlog()
    print(f"Promoted {promoted_count} tasks to READY status")

if __name__ == "__main__":
    main()