"""Mutation testing configuration for Docker Optimizer Agent.

This configuration file controls how mutmut performs mutation testing
to evaluate the quality of our test suite.
"""

def pre_mutation(context):
    """Called before each mutation is applied."""
    # Skip mutations in test files and __init__.py files
    if 'test_' in context.filename or '__init__.py' in context.filename:
        context.skip = True
    
    # Skip mutations in specific files that are hard to test
    skip_files = [
        'cli.py',  # CLI interactions are complex to test thoroughly
        'external_security.py',  # External tool integration
    ]
    
    if any(skip_file in context.filename for skip_file in skip_files):
        context.skip = True


def post_mutation(context):
    """Called after each mutation is applied and tested."""
    pass


# Mutation testing targets
targets = [
    'src/docker_optimizer/optimizer.py',
    'src/docker_optimizer/parser.py',
    'src/docker_optimizer/security.py',
    'src/docker_optimizer/performance.py',
    'src/docker_optimizer/models.py',
    'src/docker_optimizer/multistage.py',
    'src/docker_optimizer/language_optimizer.py',
]

# Test command to run
runner = 'pytest tests/test_optimizer.py tests/test_parser.py tests/test_security.py tests/test_performance.py tests/test_models.py tests/test_multistage.py tests/test_language_optimizer.py -x --tb=no -q'

# Directories to include/exclude
paths_to_mutate = 'src/docker_optimizer/'
tests_dir = 'tests/'

# Backup directory
backup_dir = '.mutmut-cache'