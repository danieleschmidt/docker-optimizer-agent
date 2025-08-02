#!/bin/bash
# Automation scripts for Docker Optimizer Agent
# Collection of utility scripts for maintenance and automation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Update all dependencies
update_dependencies() {
    log_info "Updating Python dependencies..."
    
    # Update pip
    python -m pip install --upgrade pip
    
    # Update development dependencies
    pip install --upgrade -e ".[dev,security]"
    
    # Update Node.js dependencies if package.json exists
    if [[ -f "package.json" ]]; then
        log_info "Updating Node.js dependencies..."
        npm update
    fi
    
    # Update pre-commit hooks
    if command -v pre-commit &> /dev/null; then
        pre-commit autoupdate
    fi
    
    log_success "Dependencies updated successfully"
}

# Run comprehensive code quality checks
quality_check() {
    log_info "Running comprehensive code quality checks..."
    
    local exit_code=0
    
    # Format code
    log_info "Formatting code..."
    black src/ tests/ || exit_code=1
    isort src/ tests/ || exit_code=1
    
    # Lint code
    log_info "Running linting..."
    ruff check src/ tests/ || exit_code=1
    
    # Type checking
    log_info "Running type checking..."
    mypy --ignore-missing-imports src/docker_optimizer/ || exit_code=1
    
    # Security scanning
    log_info "Running security scans..."
    bandit -r src/ -f json -o dist/security-report.json || log_warning "Security issues found"
    safety check --json --output dist/safety-report.json || log_warning "Dependency vulnerabilities found"
    
    # Run tests with coverage
    log_info "Running tests with coverage..."
    pytest tests/ --cov=docker_optimizer --cov-report=html --cov-fail-under=85 || exit_code=1
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "All quality checks passed"
    else
        log_error "Some quality checks failed"
    fi
    
    return $exit_code
}

# Clean up repository
cleanup_repo() {
    log_info "Cleaning up repository..."
    
    # Remove Python cache files
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    
    # Remove build artifacts
    rm -rf build/ dist/ *.egg-info/
    
    # Remove test artifacts
    rm -rf .pytest_cache/ htmlcov/ .coverage coverage.xml
    rm -rf .mypy_cache/ .ruff_cache/
    
    # Remove Node.js artifacts if they exist
    rm -rf node_modules/ npm-debug.log* yarn-debug.log* yarn-error.log*
    
    # Remove temporary files
    find . -type f -name "*.tmp" -delete 2>/dev/null || true
    find . -type f -name "*.log" -delete 2>/dev/null || true
    
    # Docker cleanup
    if command -v docker &> /dev/null; then
        log_info "Cleaning up Docker artifacts..."
        docker system prune -f --volumes || log_warning "Docker cleanup failed"
    fi
    
    log_success "Repository cleanup completed"
}

# Update documentation
update_docs() {
    log_info "Updating documentation..."
    
    # Generate API documentation
    if command -v pydoc &> /dev/null; then
        mkdir -p docs/generated
        python -m pydoc -w docker_optimizer
        mv *.html docs/generated/ 2>/dev/null || true
    fi
    
    # Update README badges and stats
    if [[ -f "scripts/update-readme-stats.py" ]]; then
        python scripts/update-readme-stats.py
    fi
    
    # Format markdown files
    if command -v prettier &> /dev/null; then
        prettier --write "**/*.md" || log_warning "Prettier formatting failed"
    fi
    
    log_success "Documentation updated"
}

# Backup important files
backup_repo() {
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    
    log_info "Creating backup in $backup_dir..."
    
    mkdir -p "$backup_dir"
    
    # Backup important configuration files
    local important_files=(
        "pyproject.toml"
        "package.json"
        "Dockerfile"
        "docker-compose.yml"
        ".env.example"
        "Makefile"
        ".github/project-metrics.json"
    )
    
    for file in "${important_files[@]}"; do
        if [[ -f "$file" ]]; then
            cp "$file" "$backup_dir/"
        fi
    done
    
    # Backup configuration directories
    local important_dirs=(
        ".github"
        "docs"
        "monitoring"
        "scripts"
    )
    
    for dir in "${important_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            cp -r "$dir" "$backup_dir/"
        fi
    done
    
    log_success "Backup created in $backup_dir"
}

# Check repository health
health_check() {
    log_info "Running repository health check..."
    
    local issues=0
    
    # Check if required files exist
    local required_files=(
        "pyproject.toml"
        "README.md"
        "LICENSE"
        "CHANGELOG.md"
        "Dockerfile"
        ".gitignore"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_warning "Missing required file: $file"
            ((issues++))
        fi
    done
    
    # Check Git status
    if [[ -d ".git" ]]; then
        local git_status
        git_status=$(git status --porcelain)
        if [[ -n "$git_status" ]]; then
            log_warning "Repository has uncommitted changes"
            ((issues++))
        fi
    fi
    
    # Check Python syntax
    if ! python -m py_compile src/docker_optimizer/*.py; then
        log_error "Python syntax errors found"
        ((issues++))
    fi
    
    # Check Docker build
    if command -v docker &> /dev/null; then
        if ! docker build --target production -t health-check . &> /dev/null; then
            log_error "Docker build failed"
            ((issues++))
        else
            docker rmi health-check &> /dev/null || true
        fi
    fi
    
    if [[ $issues -eq 0 ]]; then
        log_success "Repository health check passed"
    else
        log_warning "Repository health check found $issues issues"
    fi
    
    return $issues
}

# Generate release notes
generate_release_notes() {
    local version=${1:-"$(python -c 'import toml; print(toml.load("pyproject.toml")["project"]["version"])')"}
    local output_file="RELEASE_NOTES_${version}.md"
    
    log_info "Generating release notes for version $version..."
    
    # Get commits since last release
    local last_tag
    last_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
    
    {
        echo "# Release Notes - Version $version"
        echo ""
        echo "Generated on: $(date)"
        echo ""
        
        if [[ -n "$last_tag" ]]; then
            echo "## Changes since $last_tag"
            echo ""
            git log "$last_tag"..HEAD --pretty=format:"- %s (%h)" --no-merges
        else
            echo "## All Changes"
            echo ""
            git log --pretty=format:"- %s (%h)" --no-merges
        fi
        
        echo ""
        echo ""
        echo "## Metrics"
        
        # Include current metrics if available
        if [[ -f ".github/project-metrics.json" ]]; then
            echo ""
            echo "### Code Quality"
            python -c "
import json
with open('.github/project-metrics.json') as f:
    data = json.load(f)
    if 'quality_metrics' in data:
        q = data['quality_metrics']
        if 'code_coverage' in q:
            print(f'- Code coverage: {q[\"code_coverage\"][\"current\"]}%')
        if 'test_metrics' in q:
            t = q['test_metrics']
            print(f'- Total tests: {t.get(\"total_tests\", \"N/A\")}')
            print(f'- Test success rate: {t.get(\"test_success_rate\", \"N/A\")}%')
" 2>/dev/null || echo "- Metrics not available"
        fi
        
    } > "$output_file"
    
    log_success "Release notes generated: $output_file"
}

# Print usage information
usage() {
    cat << EOF
Docker Optimizer Agent - Automation Scripts

Usage: $0 <command> [options]

Commands:
    update-deps         Update all dependencies
    quality-check       Run comprehensive code quality checks
    cleanup            Clean up repository artifacts
    update-docs        Update documentation
    backup             Create backup of important files
    health-check       Run repository health check
    release-notes      Generate release notes [version]
    collect-metrics    Collect project metrics
    help               Show this help message

Examples:
    $0 quality-check
    $0 update-deps
    $0 release-notes v1.2.0
    $0 cleanup

EOF
}

# Main function
main() {
    case "${1:-help}" in
        "update-deps")
            update_dependencies
            ;;
        "quality-check")
            quality_check
            ;;
        "cleanup")
            cleanup_repo
            ;;
        "update-docs")
            update_docs
            ;;
        "backup")
            backup_repo
            ;;
        "health-check")
            health_check
            ;;
        "release-notes")
            generate_release_notes "${2:-}"
            ;;
        "collect-metrics")
            python scripts/collect-metrics.py --summary
            ;;
        "help"|*)
            usage
            ;;
    esac
}

# Run main function with all arguments
main "$@"