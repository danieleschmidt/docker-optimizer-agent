#!/usr/bin/env python3
"""Production Deployment Script for Docker Optimizer Agent."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docker_optimizer.production_readiness import (
    ProductionReadinessAssessment,
    ReadinessLevel,
    assess_production_readiness,
    generate_deployment_checklist
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main deployment preparation workflow."""
    parser = argparse.ArgumentParser(description="Docker Optimizer Production Deployment")
    parser.add_argument(
        "--level",
        choices=["development", "staging", "production", "enterprise"],
        default="production",
        help="Target readiness level (default: production)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("deployment_reports"),
        help="Output directory for reports"
    )
    parser.add_argument(
        "--checklist",
        action="store_true",
        help="Generate deployment checklist"
    )
    parser.add_argument(
        "--json-report",
        action="store_true",
        help="Generate JSON report"
    )
    parser.add_argument(
        "--fix-issues",
        action="store_true",
        help="Attempt to auto-fix detected issues"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert level string to enum
    target_level = ReadinessLevel(args.level)
    
    print(f"\n🚀 Docker Optimizer Production Deployment Assessment")
    print(f"Target Level: {target_level.value.upper()}")
    print(f"Assessment Time: {datetime.now().isoformat()}")
    print("=" * 60)
    
    try:
        # Run production readiness assessment
        print("\n📊 Running production readiness assessment...")
        report = assess_production_readiness(target_level)
        
        # Display summary
        print(f"\n📋 ASSESSMENT SUMMARY")
        print(f"Overall Status: {report.overall_status.value.upper()}")
        print(f"Readiness Score: {report.score:.1f}/100")
        print(f"Total Checks: {len(report.checks)}")
        
        # Count by status
        pass_count = len([c for c in report.checks if c.status.value == "pass"])
        warn_count = len([c for c in report.checks if c.status.value == "warning"])
        fail_count = len([c for c in report.checks if c.status.value == "critical"])
        skip_count = len([c for c in report.checks if c.status.value == "skipped"])
        
        print(f"✅ Passed: {pass_count}")
        print(f"⚠️  Warnings: {warn_count}")
        print(f"❌ Critical: {fail_count}")
        print(f"⏭️  Skipped: {skip_count}")
        
        # Show critical issues first
        critical_issues = [c for c in report.checks if c.status.value == "critical"]
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(critical_issues)}):")
            for issue in critical_issues:
                print(f"   ❌ {issue.name}: {issue.message}")
                if issue.recommendation:
                    print(f"      → {issue.recommendation}")
                if issue.fix_command:
                    print(f"      💻 {issue.fix_command}")
        
        # Show warnings
        warnings = [c for c in report.checks if c.status.value == "warning"]
        if warnings and len(warnings) <= 10:  # Show up to 10 warnings
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for warning in warnings[:10]:
                print(f"   ⚠️  {warning.name}: {warning.message}")
                if warning.recommendation:
                    print(f"      → {warning.recommendation}")
        elif len(warnings) > 10:
            print(f"\n⚠️  WARNINGS ({len(warnings)} total, showing first 10):")
            for warning in warnings[:10]:
                print(f"   ⚠️  {warning.name}: {warning.message}")
            print(f"   ... and {len(warnings) - 10} more warnings")
        
        # Auto-fix issues if requested
        if args.fix_issues:
            print(f"\n🔧 ATTEMPTING AUTO-FIXES...")
            fixed_count = attempt_auto_fixes(report)
            if fixed_count > 0:
                print(f"✅ Applied {fixed_count} automatic fixes")
                print("ℹ️  Re-run assessment to see updated status")
            else:
                print("ℹ️  No automatic fixes available")
        
        # Generate outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        if args.json_report:
            json_path = args.output_dir / f"readiness_report_{target_level.value}_{timestamp}.json"
            assessor = ProductionReadinessAssessment()
            assessor.export_report(report, json_path)
            print(f"\n📄 JSON report saved: {json_path}")
        
        # Deployment checklist
        if args.checklist:
            checklist_path = args.output_dir / f"deployment_checklist_{target_level.value}_{timestamp}.txt"
            checklist = generate_deployment_checklist(report)
            
            with open(checklist_path, 'w') as f:
                f.write(f"Docker Optimizer Deployment Checklist\n")
                f.write(f"Target Level: {target_level.value.upper()}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n\n")
                f.write("\n".join(checklist))
            
            print(f"📋 Deployment checklist saved: {checklist_path}")
        
        # Generate deployment readiness verdict
        print(f"\n🎯 DEPLOYMENT READINESS VERDICT:")
        if report.overall_status.value == "pass":
            print("✅ READY FOR DEPLOYMENT")
            print(f"   System meets {target_level.value} readiness requirements")
        elif report.overall_status.value == "warning":
            print("⚠️  DEPLOYMENT WITH CAUTION")
            print(f"   System has warnings but meets minimum {target_level.value} requirements")
            print("   Address warnings for optimal production operation")
        else:
            print("❌ NOT READY FOR DEPLOYMENT")
            print(f"   Critical issues must be resolved before {target_level.value} deployment")
            if critical_issues:
                print(f"   Resolve {len(critical_issues)} critical issue(s) first")
        
        # Recommendations summary
        if report.deployment_recommendations:
            print(f"\n📋 KEY DEPLOYMENT RECOMMENDATIONS:")
            for rec in report.deployment_recommendations[:5]:  # Top 5
                print(f"   • {rec}")
        
        if report.security_recommendations:
            print(f"\n🔒 SECURITY RECOMMENDATIONS:")
            for rec in report.security_recommendations[:3]:  # Top 3
                print(f"   • {rec}")
        
        # Exit with appropriate code
        if report.overall_status.value == "critical":
            sys.exit(1)  # Critical issues
        elif report.overall_status.value == "warning":
            sys.exit(2)  # Warnings present
        else:
            sys.exit(0)  # All good
            
    except Exception as e:
        logger.error(f"Deployment assessment failed: {e}")
        print(f"\n❌ DEPLOYMENT ASSESSMENT FAILED: {e}")
        sys.exit(3)


def attempt_auto_fixes(report) -> int:
    """Attempt to automatically fix detected issues."""
    fixed_count = 0
    
    for check in report.checks:
        if check.status.value == "critical" and check.fix_command:
            try:
                print(f"   🔧 Fixing: {check.name}")
                
                # Handle environment variable fixes
                if check.fix_command.startswith("export "):
                    env_assignment = check.fix_command[7:]  # Remove "export "
                    if "=" in env_assignment:
                        var_name, var_value = env_assignment.split("=", 1)
                        os.environ[var_name] = var_value
                        print(f"      ✅ Set {var_name}={var_value}")
                        fixed_count += 1
                
                # Handle directory creation
                elif check.fix_command.startswith("mkdir -p "):
                    dir_path = check.fix_command[9:]  # Remove "mkdir -p "
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    print(f"      ✅ Created directory: {dir_path}")
                    fixed_count += 1
                
                # Handle pip installs (be cautious)
                elif check.fix_command.startswith("pip install "):
                    print(f"      ⚠️  Skipping pip install (requires manual approval)")
                
                else:
                    print(f"      ⚠️  Manual fix required: {check.fix_command}")
                    
            except Exception as e:
                print(f"      ❌ Fix failed: {e}")
    
    return fixed_count


if __name__ == "__main__":
    main()