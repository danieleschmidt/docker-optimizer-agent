#!/bin/bash
set -e

echo "🚀 Setting up Docker Optimizer Agent development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    jq \
    tree \
    htop \
    unzip \
    wget

# Install Trivy for security scanning
echo "🔒 Installing Trivy security scanner..."
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Install Docker Compose if not present
echo "🐳 Ensuring Docker Compose is available..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev,security,trivy]"

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p {logs,reports,cache,.pytest_cache}

# Set up environment file
echo "⚙️ Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env 2>/dev/null || cat > .env << EOF
# Docker Optimizer Agent Environment Configuration
DOCKER_OPTIMIZER_LOG_LEVEL=INFO
DOCKER_OPTIMIZER_CACHE_TTL=3600
DOCKER_OPTIMIZER_MAX_WORKERS=4
DOCKER_OPTIMIZER_ENABLE_METRICS=true
DOCKER_OPTIMIZER_TRIVY_TIMEOUT=300
EOF
fi

# Initialize git hooks
echo "🎣 Setting up git hooks..."
if [ ! -f .git/hooks/pre-commit ]; then
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Run pre-commit hooks
exec pre-commit run --all-files
EOF
    chmod +x .git/hooks/pre-commit
fi

# Run initial tests to verify setup
echo "🧪 Running initial test suite..."
python -m pytest tests/test_import_dependencies.py -v

# Install additional development tools
echo "🛠️ Installing additional development tools..."
pip install --upgrade \
    ipython \
    jupyter \
    notebook \
    sphinx \
    sphinx-rtd-theme

# Set up shell aliases
echo "📝 Setting up shell aliases..."
cat >> ~/.bashrc << 'EOF'

# Docker Optimizer Agent aliases
alias docker-opt='python -m docker_optimizer.cli'
alias run-tests='pytest tests/ --cov=docker_optimizer --cov-report=html'
alias run-lint='ruff check src/ tests/ && black --check src/ tests/ && mypy src/'
alias run-security='bandit -r src/ && safety check'
alias start-monitoring='docker-compose -f monitoring/docker-compose.yml up -d'
alias stop-monitoring='docker-compose -f monitoring/docker-compose.yml down'

# Git aliases
alias git-status='git status --short --branch'
alias git-log='git log --oneline --graph --decorate --all'
alias git-clean='git clean -fd && git reset --hard HEAD'

# Development helpers
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias tree='tree -I "__pycache__|*.pyc|.git|.pytest_cache|.mypy_cache|htmlcov"'
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  make help          - Show all available commands"
echo "  make test          - Run test suite"
echo "  make lint          - Run code quality checks"
echo "  docker-opt --help  - Show CLI help"
echo ""
echo "🔗 Useful URLs (when services are running):"
echo "  http://localhost:8000  - API Server"
echo "  http://localhost:3000  - Grafana Dashboard (admin/admin)"
echo "  http://localhost:9090  - Prometheus Metrics"
echo ""
echo "Happy coding! 🚀"