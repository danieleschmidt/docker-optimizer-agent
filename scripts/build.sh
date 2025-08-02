#!/bin/bash
# Build automation script for Docker Optimizer Agent
# This script handles building, testing, and packaging for different environments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="development"
SKIP_TESTS=false
SKIP_SECURITY=false
PUSH_IMAGES=false
REGISTRY=""
VERSION=$(python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build automation script for Docker Optimizer Agent

OPTIONS:
    -t, --type TYPE         Build type: development, testing, production, cli (default: development)
    -s, --skip-tests        Skip running tests
    -S, --skip-security     Skip security scans
    -p, --push              Push images to registry
    -r, --registry URL      Docker registry URL
    -v, --version VERSION   Override version (default: from pyproject.toml)
    -h, --help              Show this help message

EXAMPLES:
    $0                          # Build development image
    $0 -t production -p         # Build and push production image
    $0 -t testing -s            # Build testing image, skip tests
    $0 --registry gcr.io/my-project --push  # Push to custom registry

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -s|--skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -S|--skip-security)
            SKIP_SECURITY=true
            shift
            ;;
        -p|--push)
            PUSH_IMAGES=true
            shift
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

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

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local deps=("docker" "python" "pip")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "$dep is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "All dependencies are available"
}

# Set up build environment
setup_build_env() {
    log_info "Setting up build environment..."
    
    # Create build directory
    mkdir -p dist/
    
    # Set build arguments
    BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    
    export BUILD_DATE VCS_REF VERSION
    
    log_success "Build environment ready (version: $VERSION, commit: $VCS_REF)"
}

# Build Python package
build_python_package() {
    log_info "Building Python package..."
    
    # Clean previous builds
    rm -rf dist/* build/* *.egg-info/
    
    # Build package
    python -m build
    
    if [[ -f "dist/docker_optimizer_agent-${VERSION}-py3-none-any.whl" ]]; then
        log_success "Python package built successfully"
    else
        log_error "Python package build failed"
        exit 1
    fi
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests as requested"
        return
    fi
    
    log_info "Running tests..."
    
    # Run linting
    log_info "Running code quality checks..."
    ruff check src/ tests/ || log_warning "Linting issues found"
    
    # Run type checking
    mypy --ignore-missing-imports src/docker_optimizer/ || log_warning "Type checking issues found"
    
    # Run unit tests
    log_info "Running unit tests..."
    pytest tests/ -m "not integration" --cov=docker_optimizer --cov-fail-under=85
    
    # Run integration tests if Docker is available
    if docker info &> /dev/null; then
        log_info "Running integration tests..."
        pytest tests/ -m "integration" || log_warning "Integration tests failed"
    else
        log_warning "Docker not available, skipping integration tests"
    fi
    
    log_success "Tests completed"
}

# Run security scans
run_security_scans() {
    if [[ "$SKIP_SECURITY" == "true" ]]; then
        log_warning "Skipping security scans as requested"
        return
    fi
    
    log_info "Running security scans..."
    
    # Run bandit security scanner
    bandit -r src/ -f json -o dist/security-report.json || log_warning "Security issues found"
    
    # Run safety check for dependencies
    safety check --json --output dist/safety-report.json || log_warning "Dependency vulnerabilities found"
    
    log_success "Security scans completed"
}

# Build Docker images
build_docker_images() {
    log_info "Building Docker images for target: $BUILD_TYPE"
    
    local image_name="docker-optimizer"
    local registry_prefix=""
    
    if [[ -n "$REGISTRY" ]]; then
        registry_prefix="${REGISTRY%/}/"
    fi
    
    case $BUILD_TYPE in
        "development")
            docker build \
                --target development \
                --build-arg BUILD_DATE="$BUILD_DATE" \
                --build-arg VCS_REF="$VCS_REF" \
                -t "${registry_prefix}${image_name}:dev" \
                -t "${registry_prefix}${image_name}:dev-${VERSION}" \
                .
            ;;
        "testing")
            docker build \
                --target testing \
                --build-arg BUILD_DATE="$BUILD_DATE" \
                --build-arg VCS_REF="$VCS_REF" \
                -t "${registry_prefix}${image_name}:test" \
                -t "${registry_prefix}${image_name}:test-${VERSION}" \
                .
            ;;
        "production")
            docker build \
                --target production \
                --build-arg BUILD_DATE="$BUILD_DATE" \
                --build-arg VCS_REF="$VCS_REF" \
                -t "${registry_prefix}${image_name}:latest" \
                -t "${registry_prefix}${image_name}:${VERSION}" \
                .
            ;;
        "cli")
            docker build \
                --target cli \
                --build-arg BUILD_DATE="$BUILD_DATE" \
                --build-arg VCS_REF="$VCS_REF" \
                -t "${registry_prefix}${image_name}:cli" \
                -t "${registry_prefix}${image_name}:cli-${VERSION}" \
                .
            ;;
        *)
            log_error "Unknown build type: $BUILD_TYPE"
            exit 1
            ;;
    esac
    
    log_success "Docker images built successfully"
}

# Push Docker images
push_docker_images() {
    if [[ "$PUSH_IMAGES" != "true" ]]; then
        log_info "Skipping image push (not requested)"
        return
    fi
    
    if [[ -z "$REGISTRY" ]]; then
        log_warning "No registry specified, pushing to Docker Hub"
        registry_prefix=""
    else
        registry_prefix="${REGISTRY%/}/"
    fi
    
    log_info "Pushing Docker images..."
    
    local image_name="docker-optimizer"
    
    case $BUILD_TYPE in
        "development")
            docker push "${registry_prefix}${image_name}:dev"
            docker push "${registry_prefix}${image_name}:dev-${VERSION}"
            ;;
        "testing")
            docker push "${registry_prefix}${image_name}:test"
            docker push "${registry_prefix}${image_name}:test-${VERSION}"
            ;;
        "production")
            docker push "${registry_prefix}${image_name}:latest"
            docker push "${registry_prefix}${image_name}:${VERSION}"
            ;;
        "cli")
            docker push "${registry_prefix}${image_name}:cli"
            docker push "${registry_prefix}${image_name}:cli-${VERSION}"
            ;;
    esac
    
    log_success "Docker images pushed successfully"
}

# Generate build report
generate_build_report() {
    log_info "Generating build report..."
    
    local report_file="dist/build-report.json"
    
    cat > "$report_file" << EOF
{
    "build_info": {
        "version": "$VERSION",
        "build_type": "$BUILD_TYPE",
        "build_date": "$BUILD_DATE",
        "vcs_ref": "$VCS_REF",
        "registry": "$REGISTRY"
    },
    "options": {
        "skip_tests": $SKIP_TESTS,
        "skip_security": $SKIP_SECURITY,
        "push_images": $PUSH_IMAGES
    },
    "artifacts": {
        "python_package": "docker_optimizer_agent-${VERSION}-py3-none-any.whl",
        "docker_images": [
            "docker-optimizer:${BUILD_TYPE}",
            "docker-optimizer:${BUILD_TYPE}-${VERSION}"
        ]
    }
}
EOF
    
    log_success "Build report generated: $report_file"
}

# Main build process
main() {
    log_info "Starting build process for Docker Optimizer Agent"
    log_info "Build type: $BUILD_TYPE, Version: $VERSION"
    
    # Execute build steps
    check_dependencies
    setup_build_env
    build_python_package
    run_tests
    run_security_scans
    build_docker_images
    push_docker_images
    generate_build_report
    
    log_success "Build process completed successfully!"
    log_info "Artifacts available in dist/ directory"
}

# Run main function
main "$@"