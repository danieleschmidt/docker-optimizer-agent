#!/usr/bin/env python3
"""
Autonomous SDLC Execution Script

Entry point for executing the complete autonomous SDLC process with
progressive enhancement, quality gates, and production deployment readiness.

Usage:
    python autonomous_sdlc_execution.py --dockerfile Dockerfile --project myapp --stage production
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Setup Python path to find our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docker_optimizer.autonomous_sdlc_coordinator import (
    AutonomousSDLCCoordinator,
    SDLCExecutionContext,
    DeploymentStage,
    OptimizationLevel,
    ValidationLevel
)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'sdlc_execution_{int(time.time())}.log')
        ]
    )
    
    # Reduce noise from some modules
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Execute Autonomous SDLC with Progressive Enhancement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Basic execution for development
    python autonomous_sdlc_execution.py --dockerfile Dockerfile --project myapp
    
    # Production deployment with aggressive optimization
    python autonomous_sdlc_execution.py \\
        --dockerfile Dockerfile \\
        --project myapp \\
        --stage production \\
        --optimization aggressive \\
        --validation strict
        
    # Full autonomous execution with all features
    python autonomous_sdlc_execution.py \\
        --dockerfile Dockerfile \\
        --project myapp \\
        --stage production \\
        --enable-scaling \\
        --enable-performance \\
        --enable-security \\
        --verbose
        '''
    )
    
    # Required arguments
    parser.add_argument(
        '--dockerfile', '-f',
        required=True,
        help='Path to Dockerfile to optimize'
    )
    
    parser.add_argument(
        '--project', '-p',
        required=True,
        help='Project name'
    )
    
    # Optional arguments
    parser.add_argument(
        '--stage', '-s',
        choices=['development', 'testing', 'staging', 'production'],
        default='production',
        help='Target deployment stage (default: production)'
    )
    
    parser.add_argument(
        '--optimization', '-o',
        choices=['basic', 'standard', 'aggressive', 'adaptive'],
        default='adaptive',
        help='Optimization level (default: adaptive)'
    )
    
    parser.add_argument(
        '--validation', '-v',
        choices=['basic', 'standard', 'strict', 'enterprise'],
        default='strict',
        help='Validation level (default: strict)'
    )
    
    # Feature flags
    parser.add_argument(
        '--enable-scaling',
        action='store_true',
        default=True,
        help='Enable autonomous scaling (default: True)'
    )
    
    parser.add_argument(
        '--disable-scaling',
        action='store_true',
        help='Disable autonomous scaling'
    )
    
    parser.add_argument(
        '--enable-performance',
        action='store_true', 
        default=True,
        help='Enable performance optimization (default: True)'
    )
    
    parser.add_argument(
        '--disable-performance',
        action='store_true',
        help='Disable performance optimization'
    )
    
    parser.add_argument(
        '--enable-security',
        action='store_true',
        default=True,
        help='Enable security scanning (default: True)'
    )
    
    parser.add_argument(
        '--disable-security',
        action='store_true',
        help='Disable security scanning'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-out',
        help='Output file for results (default: stdout)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'yaml', 'text'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--config',
        help='Path to configuration file (JSON/YAML)'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith(('.yml', '.yaml')):
                import yaml
                return yaml.safe_load(f)
            else:
                raise ValueError("Config file must be JSON or YAML")
    except Exception as e:
        logging.error(f"Failed to load config file {config_path}: {e}")
        sys.exit(1)


def create_execution_context(args: argparse.Namespace, custom_config: dict) -> SDLCExecutionContext:
    """Create SDLC execution context from arguments."""
    # Map string arguments to enums
    stage_map = {
        'development': DeploymentStage.DEVELOPMENT,
        'testing': DeploymentStage.TESTING,
        'staging': DeploymentStage.STAGING,
        'production': DeploymentStage.PRODUCTION
    }
    
    optimization_map = {
        'basic': OptimizationLevel.BASIC,
        'standard': OptimizationLevel.STANDARD,
        'aggressive': OptimizationLevel.AGGRESSIVE,
        'adaptive': OptimizationLevel.ADAPTIVE
    }
    
    validation_map = {
        'basic': ValidationLevel.BASIC,
        'standard': ValidationLevel.STANDARD,
        'strict': ValidationLevel.STRICT,
        'enterprise': ValidationLevel.ENTERPRISE
    }
    
    # Handle disable flags
    enable_scaling = args.enable_scaling and not args.disable_scaling
    enable_performance = args.enable_performance and not args.disable_performance
    enable_security = args.enable_security and not args.disable_security
    
    return SDLCExecutionContext(
        project_name=args.project,
        dockerfile_path=args.dockerfile,
        target_stage=stage_map[args.stage],
        optimization_level=optimization_map[args.optimization],
        validation_level=validation_map[args.validation],
        enable_autonomous_scaling=enable_scaling,
        enable_performance_optimization=enable_performance,
        enable_security_scanning=enable_security,
        custom_config=custom_config
    )


def format_output(result_data: dict, format_type: str) -> str:
    """Format output data."""
    if format_type == 'json':
        return json.dumps(result_data, indent=2, default=str)
    elif format_type == 'yaml':
        import yaml
        return yaml.dump(result_data, default_flow_style=False)
    else:  # text format
        return format_text_output(result_data)


def format_text_output(result_data: dict) -> str:
    """Format result data as human-readable text."""
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("AUTONOMOUS SDLC EXECUTION RESULTS")
    lines.append("=" * 80)
    lines.append("")
    
    # Executive Summary
    lines.append("📊 EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Project: {result_data['project_name']}")
    lines.append(f"Execution ID: {result_data['execution_id']}")
    lines.append(f"Duration: {result_data['duration']:.2f} seconds")
    lines.append(f"Overall Success: {'✅ YES' if result_data['overall_success'] else '❌ NO'}")
    lines.append(f"Deployment Ready: {'✅ YES' if result_data['deployment_ready'] else '❌ NO'}")
    lines.append("")
    
    # Scores
    lines.append("📈 QUALITY SCORES")
    lines.append("-" * 40)
    lines.append(f"Overall Quality: {result_data['quality_score']:.1f}/100")
    lines.append(f"Performance: {result_data['performance_score']:.1f}/100")
    lines.append(f"Security: {result_data['security_score']:.1f}/100")
    lines.append("")
    
    # Phases Completed
    lines.append("🚀 SDLC PHASES COMPLETED")
    lines.append("-" * 40)
    for phase in result_data['phases_completed']:
        lines.append(f"✅ {phase.replace('_', ' ').title()}")
    lines.append("")
    
    # Enhancement Generations
    lines.append("⚡ PROGRESSIVE ENHANCEMENT GENERATIONS")
    lines.append("-" * 40)
    for generation in result_data['generations_completed']:
        gen_name = generation.replace('generation_', 'Generation ').replace('_', ' - ').title()
        lines.append(f"✅ {gen_name}")
    lines.append("")
    
    # Issues and Recommendations
    if result_data.get('issues'):
        lines.append("⚠️ ISSUES FOUND")
        lines.append("-" * 40)
        for issue in result_data['issues']:
            lines.append(f"❌ {issue}")
        lines.append("")
    
    if result_data.get('recommendations'):
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 40)
        for rec in result_data['recommendations'][:10]:  # Limit to top 10
            lines.append(f"• {rec}")
        lines.append("")
    
    # Artifacts Summary
    if result_data.get('artifacts'):
        lines.append("📦 GENERATED ARTIFACTS")
        lines.append("-" * 40)
        for artifact_type in result_data['artifacts'].keys():
            lines.append(f"📄 {artifact_type.replace('_', ' ').title()}")
        lines.append("")
    
    # Footer
    lines.append("=" * 80)
    lines.append("🎯 NEXT STEPS")
    lines.append("-" * 40)
    
    if result_data['overall_success'] and result_data['deployment_ready']:
        lines.append("✅ Ready for deployment!")
        lines.append("• Review generated Kubernetes manifests")
        lines.append("• Set up monitoring and alerting")
        lines.append("• Configure CI/CD pipeline")
    else:
        lines.append("⚠️ Address issues before deployment:")
        lines.append("• Review quality gate failures")
        lines.append("• Implement security fixes")
        lines.append("• Optimize performance bottlenecks")
    
    lines.append("")
    lines.append("🔄 For continuous improvement:")
    lines.append("• Monitor performance metrics")
    lines.append("• Review security scan results")
    lines.append("• Update optimization strategies")
    lines.append("")
    lines.append("=" * 80)
    
    return "\\n".join(lines)


async def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load custom configuration if provided
    custom_config = {}
    if args.config:
        custom_config = load_config(args.config)
    
    # Validate Dockerfile exists
    if not Path(args.dockerfile).exists():
        logger.error(f"Dockerfile not found: {args.dockerfile}")
        sys.exit(1)
    
    logger.info("🚀 Starting Autonomous SDLC Execution")
    logger.info(f"Project: {args.project}")
    logger.info(f"Dockerfile: {args.dockerfile}")
    logger.info(f"Target Stage: {args.stage}")
    logger.info(f"Optimization Level: {args.optimization}")
    
    try:
        # Create execution context
        context = create_execution_context(args, custom_config)
        
        # Initialize and execute SDLC coordinator
        coordinator = AutonomousSDLCCoordinator()
        result = await coordinator.execute_autonomous_sdlc(context)
        
        # Format and output results
        result_data = result.to_dict()
        formatted_output = format_output(result_data, args.format)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(formatted_output)
            logger.info(f"Results written to: {args.output}")
        else:
            print("\\n" + formatted_output)
        
        # Exit with appropriate code
        sys.exit(0 if result.overall_success else 1)
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())