"""Command-line interface for quantum task planner."""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import click
import yaml

from .core.exceptions import QuantumTaskPlannerError
from .core.planner import PlannerConfig, QuantumTaskPlanner
from .models.resource import Resource
from .models.schedule import OptimizationObjective
from .models.task import Task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool, quiet: bool):
    """Setup logging configuration."""
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


@click.group()
@click.version_option(version="0.1.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-error output")
@click.option("--config-file", type=click.Path(exists=True), help="Configuration file path")
@click.pass_context
def cli(ctx, verbose: bool, quiet: bool, config_file: Optional[str]):
    """Quantum-Inspired Task Planner - Intelligent scheduling with quantum algorithms."""
    ctx.ensure_object(dict)

    setup_logging(verbose, quiet)

    # Load configuration
    config = PlannerConfig()
    if config_file:
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith(('.yml', '.yaml')):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

                # Update config with loaded values
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)

    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet


@cli.command()
@click.option("--schedule-id", required=True, help="Unique schedule identifier")
@click.option("--name", required=True, help="Schedule name")
@click.option("--description", help="Schedule description")
@click.option("--start-time", help="Schedule start time (ISO format)")
@click.option("--objectives", multiple=True,
              type=click.Choice(['minimize_makespan', 'minimize_cost', 'maximize_throughput', 'balance_load', 'quantum_optimal']),
              help="Optimization objectives")
@click.option("--output", "-o", type=click.Path(), help="Output file for schedule")
@click.pass_context
def create_schedule(ctx, schedule_id: str, name: str, description: Optional[str],
                   start_time: Optional[str], objectives: List[str], output: Optional[str]):
    """Create a new schedule."""
    try:
        config = ctx.obj['config']

        # Parse start time
        if start_time:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        else:
            start_dt = datetime.utcnow()

        # Parse objectives
        obj_list = [OptimizationObjective(obj) for obj in objectives] if objectives else None

        with QuantumTaskPlanner(config) as planner:
            schedule = planner.create_schedule(
                schedule_id=schedule_id,
                name=name,
                start_time=start_dt,
                description=description,
                objectives=obj_list
            )

            click.echo(f"✅ Schedule '{schedule_id}' created successfully")

            if output:
                schedule_data = schedule.to_dict()
                with open(output, 'w') as f:
                    json.dump(schedule_data, f, indent=2, default=str)
                click.echo(f"📄 Schedule saved to {output}")

            if ctx.obj['verbose']:
                click.echo("📊 Schedule details:")
                click.echo(f"   ID: {schedule.id}")
                click.echo(f"   Name: {schedule.name}")
                click.echo(f"   Start Time: {schedule.start_time}")
                click.echo(f"   Status: {schedule.status}")
                click.echo(f"   Objectives: {[obj.value for obj in schedule.objectives]}")

    except QuantumTaskPlannerError as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--schedule-file", required=True, type=click.Path(exists=True),
              help="Schedule definition file (JSON/YAML)")
@click.option("--task-file", type=click.Path(exists=True),
              help="Task definitions file (JSON/YAML)")
@click.option("--resource-file", type=click.Path(exists=True),
              help="Resource definitions file (JSON/YAML)")
@click.option("--algorithm", "-a",
              type=click.Choice(['quantum_annealing', 'qaoa', 'vqe_dependencies']),
              default='quantum_annealing', help="Optimization algorithm")
@click.option("--async-opt", is_flag=True, help="Run optimization asynchronously")
@click.option("--output", "-o", type=click.Path(), help="Output file for optimized schedule")
@click.option("--metrics-output", type=click.Path(), help="Output file for optimization metrics")
@click.pass_context
def optimize(ctx, schedule_file: str, task_file: Optional[str], resource_file: Optional[str],
            algorithm: str, async_opt: bool, output: Optional[str], metrics_output: Optional[str]):
    """Optimize a schedule using quantum algorithms."""
    try:
        config = ctx.obj['config']

        # Load schedule
        with open(schedule_file, 'r') as f:
            if schedule_file.endswith(('.yml', '.yaml')):
                schedule_data = yaml.safe_load(f)
            else:
                schedule_data = json.load(f)

        with QuantumTaskPlanner(config) as planner:
            # Create schedule
            schedule = planner.create_schedule(
                schedule_id=schedule_data['id'],
                name=schedule_data['name'],
                start_time=datetime.fromisoformat(schedule_data['start_time']),
                description=schedule_data.get('description'),
                objectives=[OptimizationObjective(obj) for obj in schedule_data.get('objectives', ['minimize_makespan'])]
            )

            # Load and add tasks
            if task_file:
                tasks = load_tasks_from_file(task_file)
                for task in tasks:
                    planner.add_task(schedule.id, task)
            elif 'tasks' in schedule_data:
                tasks = [Task.from_dict(task_data) for task_data in schedule_data['tasks']]
                for task in tasks:
                    planner.add_task(schedule.id, task)

            # Load and add resources
            if resource_file:
                resources = load_resources_from_file(resource_file)
                for resource in resources:
                    planner.add_resource(schedule.id, resource)
            elif 'resources' in schedule_data:
                resources = [Resource.from_dict(res_data) for res_data in schedule_data['resources']]
                for resource in resources:
                    planner.add_resource(schedule.id, resource)

            click.echo(f"🚀 Starting optimization with {algorithm} algorithm...")
            click.echo(f"📊 Tasks: {len(schedule.tasks)}, Resources: {len(schedule.resources)}")

            # Run optimization
            if async_opt:
                future = planner.optimize_schedule(schedule.id, algorithm, async_optimization=True)
                click.echo("⏳ Optimization submitted asynchronously")

                # Wait for completion with progress indication
                import time
                while not future.done():
                    click.echo(".", nl=False)
                    time.sleep(1)
                click.echo()

                optimized_schedule = future.result()
            else:
                optimized_schedule = planner.optimize_schedule(schedule.id, algorithm)

            click.echo("✅ Optimization completed successfully!")

            # Display results
            if optimized_schedule.metrics:
                metrics = optimized_schedule.metrics
                click.echo("📈 Results:")
                click.echo(f"   Makespan: {metrics.makespan}")
                click.echo(f"   Total Cost: ${metrics.total_cost:.2f}")
                click.echo(f"   Constraint Violations: {metrics.constraint_violations}")
                click.echo(f"   Optimization Time: {metrics.optimization_time}")
                click.echo(f"   Iterations: {metrics.iterations}")
                click.echo(f"   Converged: {'Yes' if metrics.convergence_achieved else 'No'}")

                # Save metrics
                if metrics_output:
                    metrics_data = {
                        'makespan_seconds': metrics.makespan.total_seconds(),
                        'total_cost': metrics.total_cost,
                        'resource_utilization': metrics.resource_utilization,
                        'constraint_violations': metrics.constraint_violations,
                        'quantum_energy': metrics.quantum_energy,
                        'optimization_time_seconds': metrics.optimization_time.total_seconds(),
                        'iterations': metrics.iterations,
                        'convergence_achieved': metrics.convergence_achieved
                    }

                    with open(metrics_output, 'w') as f:
                        json.dump(metrics_data, f, indent=2)
                    click.echo(f"📊 Metrics saved to {metrics_output}")

            # Save optimized schedule
            if output:
                schedule_data = optimized_schedule.to_dict()
                with open(output, 'w') as f:
                    json.dump(schedule_data, f, indent=2, default=str)
                click.echo(f"💾 Optimized schedule saved to {output}")

    except QuantumTaskPlannerError as e:
        click.echo(f"❌ Optimization Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during optimization: {e}")
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--schedule-id", help="Specific schedule ID to show status for")
@click.option("--format", "output_format", type=click.Choice(['table', 'json', 'yaml']),
              default='table', help="Output format")
@click.pass_context
def status(ctx, schedule_id: Optional[str], output_format: str):
    """Show schedule optimization status."""
    try:
        config = ctx.obj['config']

        with QuantumTaskPlanner(config) as planner:
            if schedule_id:
                # Show specific schedule status
                status_info = planner.get_optimization_status(schedule_id)

                if output_format == 'json':
                    click.echo(json.dumps(status_info, indent=2, default=str))
                elif output_format == 'yaml':
                    click.echo(yaml.dump(status_info, default_flow_style=False))
                else:
                    # Table format
                    click.echo(f"📋 Schedule Status: {schedule_id}")
                    click.echo(f"   Status: {status_info.get('status', 'unknown')}")
                    click.echo(f"   Created: {status_info.get('created_at', 'unknown')}")
                    click.echo(f"   Updated: {status_info.get('updated_at', 'unknown')}")
                    click.echo(f"   Tasks: {status_info.get('task_count', 0)}")
                    click.echo(f"   Resources: {status_info.get('resource_count', 0)}")
                    click.echo(f"   Assignments: {status_info.get('assignment_count', 0)}")

                    if 'metrics' in status_info:
                        metrics = status_info['metrics']
                        click.echo(f"   Makespan: {metrics.get('makespan_seconds', 0):.1f}s")
                        click.echo(f"   Total Cost: ${metrics.get('total_cost', 0):.2f}")
                        click.echo(f"   Violations: {metrics.get('constraint_violations', 0)}")
            else:
                # Show all schedules
                schedules = planner.list_schedules()

                if output_format == 'json':
                    click.echo(json.dumps(schedules, indent=2, default=str))
                elif output_format == 'yaml':
                    click.echo(yaml.dump(schedules, default_flow_style=False))
                else:
                    # Table format
                    if schedules:
                        click.echo("📋 Active Schedules:")
                        click.echo()
                        click.echo(f"{'ID':<20} {'Name':<20} {'Status':<12} {'Tasks':<6} {'Resources':<9} {'Updated':<19}")
                        click.echo("-" * 86)

                        for schedule in schedules:
                            updated = schedule['updated_at'][:19] if len(schedule['updated_at']) > 19 else schedule['updated_at']
                            click.echo(f"{schedule['id']:<20} {schedule['name'][:19]:<20} "
                                     f"{schedule['status']:<12} {schedule['task_count']:<6} "
                                     f"{schedule['resource_count']:<9} {updated:<19}")
                    else:
                        click.echo("No active schedules found.")

    except QuantumTaskPlannerError as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--format", "output_format", type=click.Choice(['table', 'json', 'yaml']),
              default='table', help="Output format")
@click.pass_context
def metrics(ctx, output_format: str):
    """Show planner performance metrics."""
    try:
        config = ctx.obj['config']

        with QuantumTaskPlanner(config) as planner:
            metrics_data = planner.get_planner_metrics()

            if output_format == 'json':
                click.echo(json.dumps(metrics_data, indent=2, default=str))
            elif output_format == 'yaml':
                click.echo(yaml.dump(metrics_data, default_flow_style=False))
            else:
                # Table format
                click.echo("📊 Planner Performance Metrics")
                click.echo()
                click.echo(f"Total Optimizations: {metrics_data.get('total_optimizations', 0)}")
                click.echo(f"Successful: {metrics_data.get('successful_optimizations', 0)}")
                click.echo(f"Failed: {metrics_data.get('failed_optimizations', 0)}")
                click.echo(f"Success Rate: {metrics_data.get('success_rate', 0.0):.1%}")
                click.echo(f"Average Time: {metrics_data.get('average_optimization_time', 0.0):.2f}s")
                click.echo()
                click.echo(f"Tasks Scheduled: {metrics_data.get('total_tasks_scheduled', 0)}")
                click.echo(f"Resources Allocated: {metrics_data.get('total_resources_allocated', 0)}")
                click.echo()

                # Algorithm usage
                if 'algorithm_usage' in metrics_data and metrics_data['algorithm_usage']:
                    click.echo("Algorithm Usage:")
                    for algo, count in metrics_data['algorithm_usage'].items():
                        click.echo(f"  {algo}: {count}")

                # Error counts
                if 'error_counts' in metrics_data and metrics_data['error_counts']:
                    click.echo()
                    click.echo("Error Counts:")
                    for error_type, count in metrics_data['error_counts'].items():
                        click.echo(f"  {error_type}: {count}")

    except QuantumTaskPlannerError as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--output", "-o", type=click.Path(), help="Output file for example schedule")
@click.option("--tasks", type=int, default=10, help="Number of example tasks")
@click.option("--resources", type=int, default=5, help="Number of example resources")
@click.option("--complexity", type=click.Choice(['simple', 'medium', 'complex']),
              default='medium', help="Schedule complexity")
def generate_example(output: Optional[str], tasks: int, resources: int, complexity: str):
    """Generate an example schedule for testing."""
    try:
        # Generate example schedule
        schedule_data = generate_example_schedule(tasks, resources, complexity)

        if output:
            with open(output, 'w') as f:
                json.dump(schedule_data, f, indent=2, default=str)
            click.echo(f"📄 Example schedule saved to {output}")
        else:
            click.echo(json.dumps(schedule_data, indent=2, default=str))

    except Exception as e:
        logger.error(f"Error generating example: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


def load_tasks_from_file(file_path: str) -> List[Task]:
    """Load tasks from file."""
    with open(file_path, 'r') as f:
        if file_path.endswith(('.yml', '.yaml')):
            tasks_data = yaml.safe_load(f)
        else:
            tasks_data = json.load(f)

    return [Task.from_dict(task_data) for task_data in tasks_data]


def load_resources_from_file(file_path: str) -> List[Resource]:
    """Load resources from file."""
    with open(file_path, 'r') as f:
        if file_path.endswith(('.yml', '.yaml')):
            resources_data = yaml.safe_load(f)
        else:
            resources_data = json.load(f)

    return [Resource.from_dict(resource_data) for resource_data in resources_data]


def generate_example_schedule(num_tasks: int, num_resources: int, complexity: str) -> Dict[str, Any]:
    """Generate example schedule data."""
    import random
    from uuid import uuid4

    # Task complexity parameters
    complexity_params = {
        'simple': {'dependencies': 0.1, 'quantum_weights': (0.5, 2.0), 'entanglement': (0.0, 0.2)},
        'medium': {'dependencies': 0.3, 'quantum_weights': (0.5, 5.0), 'entanglement': (0.0, 0.5)},
        'complex': {'dependencies': 0.5, 'quantum_weights': (1.0, 10.0), 'entanglement': (0.2, 0.8)}
    }

    params = complexity_params[complexity]

    # Generate tasks
    tasks = []
    task_ids = [f"task_{i:03d}" for i in range(num_tasks)]

    for i, task_id in enumerate(task_ids):
        # Generate dependencies
        dependencies = []
        if i > 0 and random.random() < params['dependencies']:
            num_deps = random.randint(1, min(3, i))
            dependencies = random.sample(task_ids[:i], num_deps)

        # Generate task
        task = {
            'id': task_id,
            'name': f"Task {i+1}",
            'description': f"Example task {i+1} for quantum planning",
            'duration_seconds': random.randint(1800, 14400),  # 30min to 4h
            'priority': random.choice(['low', 'medium', 'high', 'critical']),
            'dependencies': dependencies,
            'resource_requirements': {
                'cpu': random.uniform(0.1, 2.0),
                'memory': random.uniform(0.5, 4.0)
            },
            'quantum_weight': random.uniform(*params['quantum_weights']),
            'entanglement_factor': random.uniform(*params['entanglement']),
            'tags': ["example", f"complexity_{complexity}"]
        }

        tasks.append(task)

    # Generate resources
    resources = []
    resource_types = ['cpu', 'memory', 'storage', 'gpu', 'worker']

    for i in range(num_resources):
        resource_type = random.choice(resource_types)
        resource = {
            'id': f"resource_{i:03d}",
            'name': f"{resource_type.title()} Resource {i+1}",
            'type': resource_type,
            'total_capacity': random.uniform(2.0, 10.0),
            'available_capacity': random.uniform(1.0, 8.0),
            'status': 'available',
            'cost_per_unit': random.uniform(0.1, 2.0),
            'efficiency_rating': random.uniform(0.7, 1.0),
            'quantum_coherence': random.uniform(0.5, 1.0),
            'superposition_factor': random.uniform(0.0, 0.3),
            'tags': ['example', f"type_{resource_type}"]
        }

        resources.append(resource)

    # Generate schedule
    schedule_data = {
        'id': f"example_schedule_{uuid4().hex[:8]}",
        'name': f"Example {complexity.title()} Schedule",
        'description': f"Generated example schedule with {num_tasks} tasks and {num_resources} resources",
        'start_time': datetime.utcnow().isoformat(),
        'objectives': ['minimize_makespan', 'minimize_cost'],
        'tasks': tasks,
        'resources': resources
    }

    return schedule_data


if __name__ == '__main__':
    cli()
