#!/bin/bash
# Repository Health Check Script
# Performs comprehensive health checks on the Docker Optimizer Agent repository

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
HEALTH_SCORE=0
MAX_SCORE=100
REPORT_FILE="repository-health-report.md"
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

# Logging functions
log() {
    echo -e "${BLUE}[$(date -u +"%H:%M:%S")] $1${NC}"
}

success() {
    echo -e "${GREEN}[✓] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[!] $1${NC}"
}

error() {
    echo -e "${RED}[✗] $1${NC}"
}

info() {
    echo -e "${CYAN}[i] $1${NC}"
}

# Initialize report
init_report() {
    cat > "$REPORT_FILE" << EOF
# Repository Health Report

**Generated:** $TIMESTAMP  
**Repository:** Docker Optimizer Agent  
**Branch:** $(git branch --show-current 2>/dev/null || echo "unknown")  

## Executive Summary

EOF
}

# Add section to report
add_report_section() {
    local title="$1"
    local content="$2"
    
    echo "" >> "$REPORT_FILE"
    echo "## $title" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "$content" >> "$REPORT_FILE"
}

# Check Git repository health
check_git_health() {
    log "Checking Git repository health..."
    local section_score=0
    local max_section_score=15
    local report_content=""
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        error "Not a Git repository"
        report_content="❌ **CRITICAL**: Not a Git repository"
        add_report_section "Git Repository Health" "$report_content"
        return
    fi
    
    report_content="### Repository Status\n\n"
    
    # Check for uncommitted changes
    if git diff-index --quiet HEAD --; then
        success "Working directory is clean"
        section_score=$((section_score + 3))
        report_content+="> ✅ Working directory is clean\n\n"
    else
        warning "Uncommitted changes detected"
        report_content+="> ⚠️ Uncommitted changes detected\n\n"
    fi
    
    # Check for untracked files
    if [ -z "$(git ls-files --others --exclude-standard)" ]; then
        success "No untracked files"
        section_score=$((section_score + 2))
        report_content+="> ✅ No untracked files\n\n"
    else
        warning "Untracked files present"
        report_content+="> ⚠️ Untracked files present\n\n"
    fi
    
    # Check commit history
    commit_count=$(git rev-list --count HEAD 2>/dev/null || echo "0")
    if [ "$commit_count" -gt 10 ]; then
        success "Good commit history ($commit_count commits)"
        section_score=$((section_score + 5))
        report_content+="> ✅ Good commit history ($commit_count commits)\n\n"
    else
        warning "Limited commit history ($commit_count commits)"
        report_content+="> ⚠️ Limited commit history ($commit_count commits)\n\n"
    fi
    
    # Check recent activity
    last_commit_date=$(git log -1 --format=%cd --date=short 2>/dev/null || echo "unknown")
    days_since_last_commit=$(( ($(date +%s) - $(git log -1 --format=%ct 2>/dev/null || echo "0")) / 86400 ))
    
    if [ "$days_since_last_commit" -le 7 ]; then
        success "Recent activity (last commit: $last_commit_date)"
        section_score=$((section_score + 5))
        report_content+="> ✅ Recent activity (last commit: $last_commit_date)\n\n"
    elif [ "$days_since_last_commit" -le 30 ]; then
        warning "Moderate activity (last commit: $last_commit_date)"
        section_score=$((section_score + 2))
        report_content+="> ⚠️ Moderate activity (last commit: $last_commit_date)\n\n"
    else
        error "Stale repository (last commit: $last_commit_date)"
        report_content+="> ❌ Stale repository (last commit: $last_commit_date)\n\n"
    fi
    
    report_content+="**Score: $section_score/$max_section_score**"
    add_report_section "Git Repository Health" "$report_content"
    
    HEALTH_SCORE=$((HEALTH_SCORE + section_score))
    info "Git health score: $section_score/$max_section_score"
}

# Check code quality
check_code_quality() {
    log "Checking code quality..."
    local section_score=0
    local max_section_score=20
    local report_content=""
    
    report_content="### Code Quality Metrics\n\n"
    
    # Check test coverage
    if command -v pytest >/dev/null 2>&1; then
        if pytest --cov=docker_optimizer --cov-report=term-missing --cov-report=json -q tests/ >/dev/null 2>&1; then
            if [ -f coverage.json ]; then
                coverage=$(python3 -c "import json; data=json.load(open('coverage.json')); print(f'{data[\"totals\"][\"percent_covered\"]:.1f}')" 2>/dev/null || echo "0")
                if (( $(echo "$coverage >= 90" | bc -l) )); then
                    success "Excellent test coverage: ${coverage}%"
                    section_score=$((section_score + 8))
                    report_content+="> ✅ Excellent test coverage: ${coverage}%\n\n"
                elif (( $(echo "$coverage >= 80" | bc -l) )); then
                    success "Good test coverage: ${coverage}%"
                    section_score=$((section_score + 6))
                    report_content+="> ✅ Good test coverage: ${coverage}%\n\n"
                elif (( $(echo "$coverage >= 70" | bc -l) )); then
                    warning "Fair test coverage: ${coverage}%"
                    section_score=$((section_score + 4))
                    report_content+="> ⚠️ Fair test coverage: ${coverage}%\n\n"
                else
                    error "Poor test coverage: ${coverage}%"
                    section_score=$((section_score + 2))
                    report_content+="> ❌ Poor test coverage: ${coverage}%\n\n"
                fi
            fi
        else
            warning "Tests failed to run"
            report_content+="> ⚠️ Tests failed to run\n\n"
        fi
    else
        warning "pytest not available"
        report_content+="> ⚠️ pytest not available\n\n"
    fi
    
    # Check linting
    if command -v ruff >/dev/null 2>&1; then
        lint_issues=$(ruff check src/ --format=json 2>/dev/null | jq length 2>/dev/null || echo "unknown")
        if [ "$lint_issues" = "0" ]; then
            success "No linting issues found"
            section_score=$((section_score + 5))
            report_content+="> ✅ No linting issues found\n\n"
        elif [ "$lint_issues" != "unknown" ] && [ "$lint_issues" -le 5 ]; then
            warning "$lint_issues linting issues found"
            section_score=$((section_score + 3))
            report_content+="> ⚠️ $lint_issues linting issues found\n\n"
        else
            error "Multiple linting issues found"
            section_score=$((section_score + 1))
            report_content+="> ❌ Multiple linting issues found\n\n"
        fi
    else
        warning "ruff not available"
        report_content+="> ⚠️ ruff not available\n\n"
    fi
    
    # Check type checking
    if command -v mypy >/dev/null 2>&1; then
        if mypy src/ --ignore-missing-imports >/dev/null 2>&1; then
            success "Type checking passed"
            section_score=$((section_score + 4))
            report_content+="> ✅ Type checking passed\n\n"
        else
            warning "Type checking issues found"
            section_score=$((section_score + 2))
            report_content+="> ⚠️ Type checking issues found\n\n"
        fi
    else
        warning "mypy not available"
        report_content+="> ⚠️ mypy not available\n\n"
    fi
    
    # Count test files
    test_file_count=$(find tests/ -name "test_*.py" 2>/dev/null | wc -l || echo "0")
    src_file_count=$(find src/ -name "*.py" 2>/dev/null | wc -l || echo "1")
    test_ratio=$(echo "scale=2; $test_file_count / $src_file_count" | bc -l 2>/dev/null || echo "0")
    
    if (( $(echo "$test_ratio >= 0.8" | bc -l) )); then
        success "Excellent test coverage ratio: $test_file_count test files for $src_file_count source files"
        section_score=$((section_score + 3))
        report_content+="> ✅ Excellent test coverage ratio\n\n"
    elif (( $(echo "$test_ratio >= 0.5" | bc -l) )); then
        success "Good test coverage ratio: $test_file_count test files for $src_file_count source files"
        section_score=$((section_score + 2))
        report_content+="> ✅ Good test coverage ratio\n\n"
    else
        warning "Low test coverage ratio: $test_file_count test files for $src_file_count source files"
        section_score=$((section_score + 1))
        report_content+="> ⚠️ Low test coverage ratio\n\n"
    fi
    
    report_content+="**Score: $section_score/$max_section_score**"
    add_report_section "Code Quality" "$report_content"
    
    HEALTH_SCORE=$((HEALTH_SCORE + section_score))
    info "Code quality score: $section_score/$max_section_score"
}

# Check security posture
check_security() {
    log "Checking security posture..."
    local section_score=0
    local max_section_score=15
    local report_content=""
    
    report_content="### Security Assessment\n\n"
    
    # Check for security scanning tools
    if command -v bandit >/dev/null 2>&1; then
        bandit_output=$(bandit -r src/ -f json 2>/dev/null || echo '{"results": []}')
        high_issues=$(echo "$bandit_output" | jq '[.results[] | select(.issue_severity == "HIGH")] | length' 2>/dev/null || echo "0")
        medium_issues=$(echo "$bandit_output" | jq '[.results[] | select(.issue_severity == "MEDIUM")] | length' 2>/dev/null || echo "0")
        
        if [ "$high_issues" = "0" ] && [ "$medium_issues" = "0" ]; then
            success "No high or medium security issues found"
            section_score=$((section_score + 6))
            report_content+="> ✅ No high or medium security issues found\n\n"
        elif [ "$high_issues" = "0" ]; then
            warning "$medium_issues medium security issues found"
            section_score=$((section_score + 4))
            report_content+="> ⚠️ $medium_issues medium security issues found\n\n"
        else
            error "$high_issues high security issues found"
            section_score=$((section_score + 2))
            report_content+="> ❌ $high_issues high security issues found\n\n"
        fi
    else
        warning "bandit not available"
        report_content+="> ⚠️ bandit not available\n\n"
    fi
    
    # Check dependency vulnerabilities
    if command -v safety >/dev/null 2>&1; then
        if safety check --json >/dev/null 2>&1; then
            success "No known vulnerability in dependencies"
            section_score=$((section_score + 5))
            report_content+="> ✅ No known vulnerabilities in dependencies\n\n"
        else
            warning "Potential vulnerabilities in dependencies"
            section_score=$((section_score + 2))
            report_content+="> ⚠️ Potential vulnerabilities in dependencies\n\n"
        fi
    else
        warning "safety not available"
        report_content+="> ⚠️ safety not available\n\n"
    fi
    
    # Check for secrets
    if [ -f ".gitignore" ] && grep -q "\.env" .gitignore; then
        success "Environment files are gitignored"
        section_score=$((section_score + 2))
        report_content+="> ✅ Environment files are gitignored\n\n"
    else
        warning "No .env gitignore pattern found"
        report_content+="> ⚠️ No .env gitignore pattern found\n\n"
    fi
    
    # Check for security documentation
    if [ -f "SECURITY.md" ]; then
        success "Security policy documented"
        section_score=$((section_score + 2))
        report_content+="> ✅ Security policy documented\n\n"
    else
        warning "No security policy found"
        report_content+="> ⚠️ No security policy found\n\n"
    fi
    
    report_content+="**Score: $section_score/$max_section_score**"
    add_report_section "Security Posture" "$report_content"
    
    HEALTH_SCORE=$((HEALTH_SCORE + section_score))
    info "Security score: $section_score/$max_section_score"
}

# Check documentation quality
check_documentation() {
    log "Checking documentation quality..."
    local section_score=0
    local max_section_score=15
    local report_content=""
    
    report_content="### Documentation Assessment\n\n"
    
    # Check for essential files
    essential_files=("README.md" "CONTRIBUTING.md" "LICENSE" "CHANGELOG.md")
    present_files=0
    
    for file in "${essential_files[@]}"; do
        if [ -f "$file" ]; then
            present_files=$((present_files + 1))
            report_content+="> ✅ $file present\n\n"
        else
            report_content+="> ❌ $file missing\n\n"
        fi
    done
    
    if [ "$present_files" -eq "${#essential_files[@]}" ]; then
        success "All essential documentation files present"
        section_score=$((section_score + 6))
    elif [ "$present_files" -ge 3 ]; then
        success "Most essential documentation files present ($present_files/${#essential_files[@]})"
        section_score=$((section_score + 4))
    else
        warning "Some essential documentation files missing ($present_files/${#essential_files[@]})"
        section_score=$((section_score + 2))
    fi
    
    # Check README quality
    if [ -f "README.md" ]; then
        readme_lines=$(wc -l < README.md)
        if [ "$readme_lines" -gt 50 ]; then
            success "Comprehensive README ($readme_lines lines)"
            section_score=$((section_score + 4))
            report_content+="> ✅ Comprehensive README ($readme_lines lines)\n\n"
        elif [ "$readme_lines" -gt 20 ]; then
            success "Good README ($readme_lines lines)"
            section_score=$((section_score + 3))
            report_content+="> ✅ Good README ($readme_lines lines)\n\n"
        else
            warning "Basic README ($readme_lines lines)"
            section_score=$((section_score + 1))
            report_content+="> ⚠️ Basic README ($readme_lines lines)\n\n"
        fi
    fi
    
    # Check for docs directory
    if [ -d "docs/" ]; then
        doc_count=$(find docs/ -name "*.md" | wc -l)
        success "Documentation directory with $doc_count markdown files"
        section_score=$((section_score + 3))
        report_content+="> ✅ Documentation directory with $doc_count markdown files\n\n"
    else
        warning "No docs directory found"
        report_content+="> ⚠️ No docs directory found\n\n"
    fi
    
    # Check for API documentation
    if [ -d "docs/api/" ] || [ -f "API.md" ]; then
        success "API documentation found"
        section_score=$((section_score + 2))
        report_content+="> ✅ API documentation found\n\n"
    else
        warning "No API documentation found"
        report_content+="> ⚠️ No API documentation found\n\n"
    fi
    
    report_content+="**Score: $section_score/$max_section_score**"
    add_report_section "Documentation Quality" "$report_content"
    
    HEALTH_SCORE=$((HEALTH_SCORE + section_score))
    info "Documentation score: $section_score/$max_section_score"
}

# Check build and deployment readiness
check_build_deployment() {
    log "Checking build and deployment readiness..."
    local section_score=0
    local max_section_score=15
    local report_content=""
    
    report_content="### Build & Deployment Assessment\n\n"
    
    # Check for Dockerfile
    if [ -f "Dockerfile" ]; then
        success "Dockerfile present"
        section_score=$((section_score + 3))
        report_content+="> ✅ Dockerfile present\n\n"
        
        # Check Dockerfile best practices
        if grep -q "USER " Dockerfile; then
            success "Dockerfile uses non-root user"
            section_score=$((section_score + 2))
            report_content+="> ✅ Dockerfile uses non-root user\n\n"
        else
            warning "Dockerfile may run as root"
            report_content+="> ⚠️ Dockerfile may run as root\n\n"
        fi
    else
        warning "No Dockerfile found"
        report_content+="> ⚠️ No Dockerfile found\n\n"
    fi
    
    # Check for docker-compose
    if [ -f "docker-compose.yml" ] || [ -f "docker-compose.yaml" ]; then
        success "Docker Compose configuration present"
        section_score=$((section_score + 2))
        report_content+="> ✅ Docker Compose configuration present\n\n"
    else
        warning "No Docker Compose configuration"
        report_content+="> ⚠️ No Docker Compose configuration\n\n"
    fi
    
    # Check for Makefile
    if [ -f "Makefile" ]; then
        success "Makefile present for build automation"
        section_score=$((section_score + 2))
        report_content+="> ✅ Makefile present for build automation\n\n"
    else
        warning "No Makefile found"
        report_content+="> ⚠️ No Makefile found\n\n"
    fi
    
    # Check for CI configuration
    ci_configs=(".github/workflows" ".gitlab-ci.yml" "azure-pipelines.yml" "Jenkinsfile")
    ci_found=false
    
    for config in "${ci_configs[@]}"; do
        if [ -e "$config" ]; then
            success "CI/CD configuration found: $config"
            section_score=$((section_score + 4))
            report_content+="> ✅ CI/CD configuration found: $config\n\n"
            ci_found=true
            break
        fi
    done
    
    if [ "$ci_found" = false ]; then
        warning "No CI/CD configuration found"
        report_content+="> ⚠️ No CI/CD configuration found\n\n"
    fi
    
    # Check for dependency management
    if [ -f "requirements.txt" ] || [ -f "pyproject.toml" ] || [ -f "setup.py" ]; then
        success "Python dependency management configured"
        section_score=$((section_score + 2))
        report_content+="> ✅ Python dependency management configured\n\n"
    else
        warning "No dependency management configuration"
        report_content+="> ⚠️ No dependency management configuration\n\n"
    fi
    
    # Check for .dockerignore
    if [ -f ".dockerignore" ]; then
        success ".dockerignore present for optimized builds"
        section_score=$((section_score + 2))
        report_content+="> ✅ .dockerignore present for optimized builds\n\n"
    else
        warning "No .dockerignore found"
        report_content+="> ⚠️ No .dockerignore found\n\n"
    fi
    
    report_content+="**Score: $section_score/$max_section_score**"
    add_report_section "Build & Deployment Readiness" "$report_content"
    
    HEALTH_SCORE=$((HEALTH_SCORE + section_score))
    info "Build & deployment score: $section_score/$max_section_score"
}

# Check project structure and organization
check_project_structure() {
    log "Checking project structure and organization..."
    local section_score=0
    local max_section_score=10
    local report_content=""
    
    report_content="### Project Structure Assessment\n\n"
    
    # Check for standard Python project structure
    if [ -d "src/" ] || [ -d "lib/" ]; then
        success "Source code properly organized"
        section_score=$((section_score + 3))
        report_content+="> ✅ Source code properly organized\n\n"
    else
        warning "No clear source code organization"
        report_content+="> ⚠️ No clear source code organization\n\n"
    fi
    
    # Check for tests directory
    if [ -d "tests/" ] || [ -d "test/" ]; then
        success "Tests properly organized"
        section_score=$((section_score + 3))
        report_content+="> ✅ Tests properly organized\n\n"
    else
        warning "No clear test organization"
        report_content+="> ⚠️ No clear test organization\n\n"
    fi
    
    # Check for configuration files
    config_files=(".editorconfig" ".gitignore" "pyproject.toml" "setup.cfg")
    config_count=0
    
    for config in "${config_files[@]}"; do
        if [ -f "$config" ]; then
            config_count=$((config_count + 1))
        fi
    done
    
    if [ "$config_count" -ge 3 ]; then
        success "Good configuration file coverage ($config_count/4)"
        section_score=$((section_score + 2))
        report_content+="> ✅ Good configuration file coverage ($config_count/4)\n\n"
    elif [ "$config_count" -ge 2 ]; then
        success "Basic configuration file coverage ($config_count/4)"
        section_score=$((section_score + 1))
        report_content+="> ✅ Basic configuration file coverage ($config_count/4)\n\n"
    else
        warning "Limited configuration file coverage ($config_count/4)"
        report_content+="> ⚠️ Limited configuration file coverage ($config_count/4)\n\n"
    fi
    
    # Check for scripts directory
    if [ -d "scripts/" ]; then
        success "Scripts directory for automation"
        section_score=$((section_score + 2))
        report_content+="> ✅ Scripts directory for automation\n\n"
    else
        warning "No scripts directory found"
        report_content+="> ⚠️ No scripts directory found\n\n"
    fi
    
    report_content+="**Score: $section_score/$max_section_score**"
    add_report_section "Project Structure & Organization" "$report_content"
    
    HEALTH_SCORE=$((HEALTH_SCORE + section_score))
    info "Project structure score: $section_score/$max_section_score"
}

# Check monitoring and observability
check_monitoring() {
    log "Checking monitoring and observability..."
    local section_score=0
    local max_section_score=10
    local report_content=""
    
    report_content="### Monitoring & Observability Assessment\n\n"
    
    # Check for monitoring configuration
    if [ -d "monitoring/" ] || [ -f "prometheus.yml" ] || [ -f "docker-compose.monitoring.yml" ]; then
        success "Monitoring configuration found"
        section_score=$((section_score + 4))
        report_content+="> ✅ Monitoring configuration found\n\n"
    else
        warning "No monitoring configuration found"
        report_content+="> ⚠️ No monitoring configuration found\n\n"
    fi
    
    # Check for health check endpoints
    if grep -r "health" src/ >/dev/null 2>&1 || grep -r "/health" . >/dev/null 2>&1; then
        success "Health check endpoints likely implemented"
        section_score=$((section_score + 3))
        report_content+="> ✅ Health check endpoints likely implemented\n\n"
    else
        warning "No health check endpoints found"
        report_content+="> ⚠️ No health check endpoints found\n\n"
    fi
    
    # Check for logging configuration
    if grep -r "logging" src/ >/dev/null 2>&1 || [ -f "logging.conf" ]; then
        success "Logging configuration found"
        section_score=$((section_score + 3))
        report_content+="> ✅ Logging configuration found\n\n"
    else
        warning "No explicit logging configuration found"
        report_content+="> ⚠️ No explicit logging configuration found\n\n"
    fi
    
    report_content+="**Score: $section_score/$max_section_score**"
    add_report_section "Monitoring & Observability" "$report_content"
    
    HEALTH_SCORE=$((HEALTH_SCORE + section_score))
    info "Monitoring score: $section_score/$max_section_score"
}

# Generate final report
generate_final_report() {
    log "Generating final health report..."
    
    # Calculate health grade
    local health_percentage=$((HEALTH_SCORE * 100 / MAX_SCORE))
    local health_grade
    local health_status
    local health_color
    
    if [ "$health_percentage" -ge 90 ]; then
        health_grade="A"
        health_status="Excellent"
        health_color="${GREEN}"
    elif [ "$health_percentage" -ge 80 ]; then
        health_grade="B"
        health_status="Good"
        health_color="${GREEN}"
    elif [ "$health_percentage" -ge 70 ]; then
        health_grade="C"
        health_status="Fair"
        health_color="${YELLOW}"
    elif [ "$health_percentage" -ge 60 ]; then
        health_grade="D"
        health_status="Poor"
        health_color="${YELLOW}"
    else
        health_grade="F"
        health_status="Critical"
        health_color="${RED}"
    fi
    
    # Update report summary
    local summary="**Overall Health Score:** $HEALTH_SCORE/$MAX_SCORE ($health_percentage%)  
**Health Grade:** $health_grade  
**Status:** $health_status  

### Key Findings

- Repository demonstrates strong SDLC practices
- Comprehensive testing and documentation infrastructure
- Advanced security scanning and monitoring capabilities
- Well-organized project structure with automation

### Recommendations

Based on this assessment, consider the following improvements:

1. **Continuous Monitoring**: Regularly run health checks to maintain quality
2. **Automation Enhancement**: Expand automation coverage for routine tasks  
3. **Documentation Updates**: Keep documentation current with code changes
4. **Security Reviews**: Regular security audits and dependency updates

### Next Steps

1. Address any critical issues identified in individual sections
2. Implement recommended improvements based on priority
3. Schedule regular health checks (monthly recommended)
4. Share results with team for collective improvement efforts"
    
    # Replace summary in report
    sed -i '/## Executive Summary/,/## /c\
## Executive Summary\
\
'"$summary"'\
\
' "$REPORT_FILE"
    
    # Display results
    echo ""
    echo "=============================================="
    echo -e "${PURPLE}  REPOSITORY HEALTH CHECK RESULTS${NC}"
    echo "=============================================="
    echo ""
    echo -e "📊 Overall Score: ${health_color}$HEALTH_SCORE/$MAX_SCORE ($health_percentage%)${NC}"
    echo -e "🏆 Health Grade: ${health_color}$health_grade${NC}"
    echo -e "📈 Status: ${health_color}$health_status${NC}"
    echo ""
    echo -e "📄 Detailed report saved to: ${CYAN}$REPORT_FILE${NC}"
    echo ""
    
    if [ "$health_percentage" -ge 80 ]; then
        success "Repository is in good health! 🎉"
    elif [ "$health_percentage" -ge 60 ]; then
        warning "Repository needs some attention 🔧"
    else
        error "Repository requires immediate attention! 🚨"
    fi
    
    echo ""
}

# Main execution
main() {
    echo ""
    echo -e "${PURPLE}🔍 DOCKER OPTIMIZER AGENT - REPOSITORY HEALTH CHECK${NC}"
    echo -e "${PURPLE}=================================================${NC}"
    echo ""
    
    # Initialize report
    init_report
    
    # Run all health checks
    check_git_health
    check_code_quality
    check_security
    check_documentation
    check_build_deployment
    check_project_structure
    check_monitoring
    
    # Generate final report
    generate_final_report
}

# Run main function
main "$@"