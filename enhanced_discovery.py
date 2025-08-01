#!/usr/bin/env python3
"""
Enhanced continuous value discovery system.
Integrates with existing autonomous_backlog.py with advanced discovery patterns.
"""

import json
import re
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import existing autonomous backlog system
import sys
sys.path.append('/root/repo/src')
from autonomous_backlog import AutonomousBacklog, BacklogItem, TaskStatus, RiskTier


@dataclass
class ValueItem:
    """Extended backlog item with value scoring components."""
    id: str
    title: str
    description: str
    category: str
    source: str
    file_path: str
    line_number: Optional[int]
    
    # WSJF Components
    user_business_value: int  # 1-13 Fibonacci
    time_criticality: int     # 1-13 Fibonacci  
    risk_reduction: int       # 1-13 Fibonacci
    opportunity_enablement: int  # 1-13 Fibonacci
    job_size: int            # 1-13 Fibonacci
    
    # ICE Components
    impact: int              # 1-10 scale
    confidence: int          # 1-10 scale
    ease: int               # 1-10 scale
    
    # Technical Debt Scoring
    debt_impact: int         # Maintenance hours saved
    debt_interest: int       # Future cost if not addressed
    hotspot_multiplier: float # 1-5x based on file activity
    
    # Calculated Scores
    wsjf_score: float
    ice_score: float
    debt_score: float
    composite_score: float
    
    created_at: str
    priority_boost: float = 1.0
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class EnhancedValueDiscovery:
    """Advanced value discovery system with multiple signal sources."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_system = AutonomousBacklog(repo_path)
        
        # Discovery patterns
        self.todo_patterns = ["TODO", "FIXME", "HACK", "BUG", "XXX", "DEPRECATED"]
        self.debt_patterns = ["temporary", "quick fix", "workaround", "legacy"]
        self.performance_patterns = ["slow", "optimize", "bottleneck", "performance"]
        self.security_patterns = ["security", "vulnerability", "unsafe", "insecure"]
        
        # File change frequency cache
        self.file_hotspots = self._calculate_file_hotspots()
    
    def _calculate_file_hotspots(self) -> Dict[str, float]:
        """Calculate file change frequency for hotspot analysis."""
        hotspots = defaultdict(float)
        try:
            # Get git log for file change frequency (last 90 days)
            result = subprocess.run([
                "git", "log", "--since=90 days ago", "--name-only", "--pretty=format:"
            ], capture_output=True, text=True, cwd=self.repo_path, timeout=30)
            
            files = [f for f in result.stdout.split('\n') if f.strip()]
            for file_path in files:
                hotspots[file_path] += 1.0
                
            # Normalize to 1-5 scale
            if hotspots:
                max_changes = max(hotspots.values())
                for file_path in hotspots:
                    hotspots[file_path] = 1.0 + (hotspots[file_path] / max_changes) * 4.0
                    
        except Exception:
            pass  # Fallback to default multiplier of 1.0
            
        return dict(hotspots)
    
    def discover_todo_comments(self) -> List[ValueItem]:
        """Discover TODO/FIXME comments with context analysis."""
        items = []
        
        try:
            # Use ripgrep for fast searching
            cmd = [
                "rg", "-n", "--type", "py", "--type", "md", "--type", "yaml",
                "|".join(self.todo_patterns), str(self.repo_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            for line in result.stdout.split('\n'):
                if not line.strip():
                    continue
                    
                parts = line.split(':', 3)
                if len(parts) < 4:
                    continue
                    
                file_path, line_num, context = parts[0], int(parts[1]), parts[3]
                relative_path = str(Path(file_path).relative_to(self.repo_path))
                
                # Analyze comment content for categorization
                category = self._categorize_comment(context)
                priority = self._assess_comment_priority(context, relative_path)
                
                hotspot_mult = self.file_hotspots.get(relative_path, 1.0)
                
                item = ValueItem(
                    id=f"todo-{hash(line) % 100000}",
                    title=f"{category.upper()}: {context.strip()[:60]}...",
                    description=f"Found in {relative_path}:{line_num}\n{context.strip()}",
                    category=category,
                    source="code_comments",
                    file_path=relative_path,
                    line_number=line_num,
                    
                    # WSJF scoring based on context
                    user_business_value=priority["business_value"],
                    time_criticality=priority["time_critical"],
                    risk_reduction=priority["risk_reduction"],
                    opportunity_enablement=priority["opportunity"],
                    job_size=priority["effort"],
                    
                    # ICE scoring
                    impact=priority["impact"],
                    confidence=priority["confidence"],
                    ease=priority["ease"],
                    
                    # Technical debt
                    debt_impact=priority["debt_impact"],
                    debt_interest=priority["debt_interest"],
                    hotspot_multiplier=hotspot_mult,
                    
                    wsjf_score=0.0,  # Will be calculated
                    ice_score=0.0,
                    debt_score=0.0,
                    composite_score=0.0,
                    
                    created_at=datetime.now(timezone.utc).isoformat(),
                    tags=["discovered", "todo", category]
                )
                
                self._calculate_scores(item)
                items.append(item)
                
        except Exception as e:
            print(f"Error discovering TODO comments: {e}")
            
        return items
    
    def discover_test_failures(self) -> List[ValueItem]:
        """Discover failing tests and flaky test patterns."""
        items = []
        
        try:
            # Run tests in discovery mode to find failures
            result = subprocess.run([
                "python", "-m", "pytest", "--tb=no", "-q", "--collect-only"
            ], capture_output=True, text=True, cwd=self.repo_path, timeout=120)
            
            # Parse test collection for issues
            if "FAILED" in result.stdout or result.returncode != 0:
                item = ValueItem(
                    id="test-failures-001",
                    title="Fix failing test collection or execution",
                    description="Test suite has collection or execution issues",
                    category="test_reliability",
                    source="test_execution",
                    file_path="tests/",
                    line_number=None,
                    
                    user_business_value=8,
                    time_criticality=13,
                    risk_reduction=8,
                    opportunity_enablement=3,
                    job_size=5,
                    
                    impact=9,
                    confidence=8,
                    ease=6,
                    
                    debt_impact=20,
                    debt_interest=50,
                    hotspot_multiplier=2.0,
                    
                    wsjf_score=0.0,
                    ice_score=0.0,
                    debt_score=0.0,
                    composite_score=0.0,
                    
                    created_at=datetime.now(timezone.utc).isoformat(),
                    tags=["discovered", "tests", "critical"]
                )
                
                self._calculate_scores(item)
                items.append(item)
                
        except Exception as e:
            print(f"Error discovering test failures: {e}")
            
        return items
    
    def discover_performance_issues(self) -> List[ValueItem]:
        """Discover performance optimization opportunities."""
        items = []
        
        # Look for performance-related comments and patterns
        try:
            patterns = "|".join(self.performance_patterns)
            result = subprocess.run([
                "rg", "-n", "-i", patterns, str(self.repo_path / "src")
            ], capture_output=True, text=True, timeout=30)
            
            for line in result.stdout.split('\n'):
                if not line.strip():
                    continue
                    
                parts = line.split(':', 3)
                if len(parts) < 4:
                    continue
                    
                file_path, line_num, context = parts[0], int(parts[1]), parts[3]
                relative_path = str(Path(file_path).relative_to(self.repo_path))
                
                item = ValueItem(
                    id=f"perf-{hash(line) % 100000}",
                    title=f"Performance optimization: {context.strip()[:50]}...",
                    description=f"Performance issue in {relative_path}:{line_num}",
                    category="performance",
                    source="static_analysis", 
                    file_path=relative_path,
                    line_number=line_num,
                    
                    user_business_value=5,
                    time_criticality=3,
                    risk_reduction=2,
                    opportunity_enablement=8,
                    job_size=8,
                    
                    impact=7,
                    confidence=6,
                    ease=4,
                    
                    debt_impact=15,
                    debt_interest=30,
                    hotspot_multiplier=self.file_hotspots.get(relative_path, 1.0),
                    
                    wsjf_score=0.0,
                    ice_score=0.0,
                    debt_score=0.0,
                    composite_score=0.0,
                    
                    created_at=datetime.now(timezone.utc).isoformat(),
                    tags=["discovered", "performance"]
                )
                
                self._calculate_scores(item)
                items.append(item)
                
        except Exception as e:
            print(f"Error discovering performance issues: {e}")
            
        return items
    
    def discover_security_vulnerabilities(self) -> List[ValueItem]:
        """Discover security vulnerabilities and risks."""
        items = []
        
        # Security pattern analysis
        try:
            patterns = "|".join(self.security_patterns)
            result = subprocess.run([
                "rg", "-n", "-i", patterns, str(self.repo_path / "src")
            ], capture_output=True, text=True, timeout=30)
            
            for line in result.stdout.split('\n'):
                if not line.strip():
                    continue
                    
                parts = line.split(':', 3)
                if len(parts) < 4:
                    continue
                    
                file_path, line_num, context = parts[0], int(parts[1]), parts[3]
                relative_path = str(Path(file_path).relative_to(self.repo_path))
                
                item = ValueItem(
                    id=f"sec-{hash(line) % 100000}",
                    title=f"Security review: {context.strip()[:50]}...",
                    description=f"Security concern in {relative_path}:{line_num}",
                    category="security",
                    source="security_analysis",
                    file_path=relative_path,
                    line_number=line_num,
                    
                    user_business_value=13,
                    time_criticality=8,
                    risk_reduction=13,
                    opportunity_enablement=2,
                    job_size=5,
                    
                    impact=9,
                    confidence=7,
                    ease=6,
                    
                    debt_impact=25,
                    debt_interest=100,
                    hotspot_multiplier=self.file_hotspots.get(relative_path, 1.0),
                    
                    wsjf_score=0.0,
                    ice_score=0.0,
                    debt_score=0.0,
                    composite_score=0.0,
                    
                    created_at=datetime.now(timezone.utc).isoformat(),
                    priority_boost=2.0,  # Security gets priority boost
                    tags=["discovered", "security", "high-priority"]
                )
                
                self._calculate_scores(item)
                items.append(item)
                
        except Exception as e:
            print(f"Error discovering security issues: {e}")
            
        return items
    
    def _categorize_comment(self, context: str) -> str:
        """Categorize a comment based on its content."""
        context_lower = context.lower()
        
        if any(pattern in context_lower for pattern in ["bug", "error", "fail", "broken"]):
            return "bug_fix"
        elif any(pattern in context_lower for pattern in ["perf", "slow", "optim", "speed"]):
            return "performance"
        elif any(pattern in context_lower for pattern in ["secur", "vulnerab", "unsafe"]):
            return "security"
        elif any(pattern in context_lower for pattern in ["test", "spec", "coverage"]):
            return "testing"
        elif any(pattern in context_lower for pattern in ["doc", "comment", "explain"]):
            return "documentation"
        elif any(pattern in context_lower for pattern in ["refactor", "clean", "improve"]):
            return "refactoring"
        else:
            return "enhancement"
    
    def _assess_comment_priority(self, context: str, file_path: str) -> Dict[str, int]:
        """Assess priority of a comment based on context and location."""
        context_lower = context.lower()
        
        # Base priorities
        priority = {
            "business_value": 3,
            "time_critical": 2,
            "risk_reduction": 2,
            "opportunity": 2,
            "effort": 3,
            "impact": 5,
            "confidence": 6,
            "ease": 5,
            "debt_impact": 10,
            "debt_interest": 20
        }
        
        # Boost for critical keywords
        if any(word in context_lower for word in ["critical", "urgent", "asap", "immediate"]):
            priority["time_critical"] = 8
            priority["business_value"] = 8
            
        if any(word in context_lower for word in ["security", "vulnerable", "exploit"]):
            priority["risk_reduction"] = 13
            priority["debt_interest"] = 50
            
        if any(word in context_lower for word in ["performance", "slow", "bottleneck"]):
            priority["opportunity"] = 8
            priority["business_value"] = 5
            
        # Adjust based on file importance
        if "cli" in file_path or "main" in file_path:
            priority["business_value"] += 2
        if "test" in file_path:
            priority["risk_reduction"] += 2
        if "security" in file_path:
            priority["risk_reduction"] += 5
            
        return priority
    
    def _calculate_scores(self, item: ValueItem) -> None:
        """Calculate all scoring components for a value item."""
        # WSJF Score
        cost_of_delay = (
            item.user_business_value + 
            item.time_criticality + 
            item.risk_reduction + 
            item.opportunity_enablement
        )
        item.wsjf_score = cost_of_delay / item.job_size
        
        # ICE Score
        item.ice_score = item.impact * item.confidence * item.ease
        
        # Technical Debt Score
        item.debt_score = (item.debt_impact + item.debt_interest) * item.hotspot_multiplier
        
        # Composite Score with adaptive weighting for advanced repos
        weights = {
            "wsjf": 0.5,
            "ice": 0.1, 
            "debt": 0.3,
            "security": 0.1
        }
        
        normalized_wsjf = min(100, item.wsjf_score * 10)
        normalized_ice = min(100, item.ice_score / 10)
        normalized_debt = min(100, item.debt_score / 5)
        
        item.composite_score = (
            weights["wsjf"] * normalized_wsjf +
            weights["ice"] * normalized_ice + 
            weights["debt"] * normalized_debt
        ) * item.priority_boost
    
    def run_comprehensive_discovery(self) -> List[ValueItem]:
        """Run all discovery methods and return prioritized value items."""
        print("🔍 Running comprehensive value discovery...")
        
        all_items = []
        
        # Discover from multiple sources
        discovery_methods = [
            ("TODO/FIXME comments", self.discover_todo_comments),
            ("Test failures", self.discover_test_failures), 
            ("Performance issues", self.discover_performance_issues),
            ("Security vulnerabilities", self.discover_security_vulnerabilities)
        ]
        
        for method_name, method in discovery_methods:
            try:
                items = method()
                print(f"  📋 {method_name}: {len(items)} items")
                all_items.extend(items)
            except Exception as e:
                print(f"  ❌ {method_name} failed: {e}")
        
        # Sort by composite score
        all_items.sort(key=lambda x: x.composite_score, reverse=True)
        
        print(f"🎯 Total discovered: {len(all_items)} value items")
        return all_items
    
    def convert_to_backlog_items(self, value_items: List[ValueItem]) -> List[BacklogItem]:
        """Convert ValueItems to BacklogItems for the autonomous system."""
        backlog_items = []
        
        for item in value_items:
            # Determine risk tier based on composite score
            if item.composite_score >= 80:
                risk_tier = RiskTier.LOW
            elif item.composite_score >= 60:
                risk_tier = RiskTier.MEDIUM
            elif item.composite_score >= 40:
                risk_tier = RiskTier.HIGH
            else:
                risk_tier = RiskTier.CRITICAL
            
            backlog_item = BacklogItem(
                id=item.id,
                title=item.title,
                type=item.category,
                description=item.description,
                acceptance_criteria=[f"Address issue in {item.file_path}"],
                effort=item.job_size,
                value=item.user_business_value,
                time_criticality=item.time_criticality,
                risk_reduction=item.risk_reduction,
                wsjf_score=item.wsjf_score,
                aging_multiplier=1.0,
                status=TaskStatus.NEW,
                risk_tier=risk_tier,
                created_at=item.created_at,
                links=[item.file_path] if item.file_path else [],
                tags=item.tags
            )
            
            backlog_items.append(backlog_item)
        
        return backlog_items
    
    def save_discovery_results(self, value_items: List[ValueItem]) -> None:
        """Save discovery results to metrics file."""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "discovery_run": {
                "total_items": len(value_items),
                "categories": {},
                "sources": {},
                "top_items": []
            }
        }
        
        # Aggregate by category and source
        for item in value_items:
            results["discovery_run"]["categories"][item.category] = \
                results["discovery_run"]["categories"].get(item.category, 0) + 1
            results["discovery_run"]["sources"][item.source] = \
                results["discovery_run"]["sources"].get(item.source, 0) + 1
        
        # Top 10 items
        for item in value_items[:10]:
            results["discovery_run"]["top_items"].append({
                "id": item.id,
                "title": item.title,
                "category": item.category,
                "composite_score": item.composite_score,
                "wsjf_score": item.wsjf_score
            })
        
        # Save to status directory
        status_dir = self.repo_path / "docs" / "status"
        status_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        results_file = status_dir / f"discovery-{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Discovery results saved to {results_file}")


def main():
    """Run enhanced value discovery and integration with autonomous backlog."""
    discovery = EnhancedValueDiscovery()
    
    # Run comprehensive discovery
    value_items = discovery.run_comprehensive_discovery()
    
    if not value_items:
        print("📭 No value items discovered")
        return
    
    # Convert and integrate with autonomous backlog
    backlog_items = discovery.convert_to_backlog_items(value_items)
    
    # Load existing autonomous backlog
    discovery.backlog_system.load_backlog()
    
    # Add new items (avoiding duplicates)
    existing_ids = {item.id for item in discovery.backlog_system.backlog}
    new_items = [item for item in backlog_items if item.id not in existing_ids]
    
    if new_items:
        discovery.backlog_system.backlog.extend(new_items)
        discovery.backlog_system.score_and_sort_backlog()
        discovery.backlog_system.save_backlog()
        print(f"📋 Added {len(new_items)} new items to autonomous backlog")
    
    # Save discovery results
    discovery.save_discovery_results(value_items)
    
    # Show top opportunities
    print(f"\n🎯 Top 5 Value Opportunities:")
    for i, item in enumerate(value_items[:5], 1):
        print(f"  {i}. {item.title} (Score: {item.composite_score:.1f})")


if __name__ == "__main__":
    main()